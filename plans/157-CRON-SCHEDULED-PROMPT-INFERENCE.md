# 157 - Cron-Scheduled Prompt Inference

## Goal

Allow Geist's current user to save a prompt with a cron schedule, have Geist
enqueue each due occurrence durably, and run inference outside the HTTP request
on a background worker thread. Preserve the completion, chat session, retry
state, and error through the existing database-backed job queue.

The initial delivery includes the backend contract, a chat-callable scheduling
tool, and a small Schedules screen. The cron engine and execution path remain
independent of the frontend.

## What exists

- `app/services/job_queue.py` already provides a process-wide background worker,
  handler registry, retry/backoff, and handlers for `tool.call` and
  `workflow.run`.
- `app/models/database/job.py` persists queued jobs and has a `run_after` field,
  but `run_after` is only a one-time visibility delay. There is no recurring
  schedule model or service that creates future jobs.
- `app/main.py` starts and stops the job worker with FastAPI lifecycle hooks.
- The `/agent/complete_text*` routes invoke synchronous `complete_text` methods
  directly from async endpoints. They do not expose a shared service that a job
  handler can call safely.
- `UserSettingsService.create_agent_from_user_settings` and
  `app/services/agent_context_provider.py` are the closest reusable composition
  points for user-specific inference.
- `LocalAgent.complete_text` and `OnlineAgent.complete_text` persist chat history
  and return a completion carrying its `chat_id`; a scheduled run can reuse that
  behavior.
- The current worker is single-threaded. That protects stateful local model
  runners, but a long inference job would also block unrelated tool/workflow
  jobs if all kinds share the same consumer.
- Job status endpoints require authentication but currently do not scope rows to
  the authenticated user. Scheduled prompt results must not inherit that gap.
- Workflow `LLMProcessor` and `AgentProcessor` are currently mock processors, so
  scheduled inference should call the real agent path rather than route through
  workflows.

## Recommended product semantics

### Cron dialect

- Accept standard five-field cron expressions: minute, hour, day of month,
  month, and day of week.
- Support numeric values, `*`, comma-separated lists, ranges, and `/step`.
- Treat both `0` and `7` as Sunday and use traditional cron OR behavior when
  both day-of-month and day-of-week are restricted.
- Do not support seconds, years, nicknames such as `@daily`, or arbitrary code.
- Validate and calculate schedules with a small standard-library implementation
  (`datetime` and `zoneinfo`); do not add a dependency for the MVP.

### Time zones and DST

- Each schedule stores an IANA time-zone name and defaults to `UTC`.
- Persist `next_run_at`, `scheduled_for`, and job timestamps as naive UTC to
  match the repository's current database convention; convert only at the cron
  evaluation boundary.
- A nonexistent wall-clock minute during spring-forward is skipped.
- A repeated wall-clock minute during fall-back fires once, not twice.

### Missed and overlapping occurrences

- Use coalescing after downtime: enqueue at most one catch-up run for an overdue
  schedule, then calculate `next_run_at` strictly after the current time. This
  prevents a restart from flooding the queue with every missed minute.
- The MVP has one inference consumer, so scheduled inferences do not execute in
  parallel. Additional occurrences can wait durably in the queue.
- Keep online inference concurrency as a later optimization. Local MLX/Torch
  model instances are stateful and should not be shared across concurrent calls
  without runner-specific locking and load tests.

### Configuration resolution

- Store optional inference overrides on the schedule: `agent_type`, `model`,
  `runner_type`, `max_tokens`, `temperature`, `top_p`, `frequency_penalty`, and
  `presence_penalty`.
- Resolve omitted values from the owning user's settings at execution time, so a
  schedule that says "use my default" follows later settings changes.
- Copy the resolved prompt and inference configuration into the job payload when
  an occurrence is enqueued. Retries therefore execute the same request even if
  the schedule is edited afterward.
- Each run creates a new chat session initially. The resulting `chat_id` is
  returned in the job result so the user can open the generated conversation.

## Architecture

```text
prompt_schedule row (next_run_at <= now)
                 |
                 v
        PromptScheduler thread
      atomically advances schedule
       and inserts one Job row
                 |
                 v
   Job(kind="prompt.inference")
                 |
                 v
       inference worker thread
                 |
                 v
       shared InferenceService
        /                    \
  LocalAgent              OnlineAgent
        \                    /
         chat history + job result
```

The scheduler never invokes a model. Its only responsibility is converting due
schedule occurrences into durable jobs. The job handler is the only scheduled
inference entry point.

## Data model and migration

### `prompt_schedule`

Add `app/models/database/prompt_schedule.py` with:

- `prompt_schedule_id`: integer primary key.
- `user_id`: required foreign key to `geist_user.user_id`, indexed.
- `name`: required display name.
- `prompt`: required text.
- `cron_expression`: required normalized five-field expression.
- `timezone`: required IANA name, default `UTC`.
- `enabled`: boolean, default true.
- `inference_config`: portable SQLAlchemy JSON object containing only validated
  optional overrides (never credentials).
- `next_run_at`: indexed UTC datetime; null only when disabled.
- `last_enqueued_at`: UTC datetime of the last occurrence converted to a job.
- `created_at` and `updated_at`.

Add the model to `app/models/database/__init__.py` so `initdb.py` registers it.

### Schedule run provenance and ownership

Add a separate `prompt_schedule_run` table rather than altering the existing
`job` table. Geist's normal `initdb.py` path uses `create_all`, which safely
creates new tables on existing installations but does not add columns to old
tables. The run table contains `prompt_schedule_run_id`, `prompt_schedule_id`,
`user_id`, `job_id`, `trigger_type`, `scheduled_for`, and `created_at`.

Add a unique constraint on `(prompt_schedule_id, scheduled_for)` to make a cron
occurrence idempotent even if two scheduler scans race. `job_id` is unique and
links each occurrence to its queue status/result. Schedule deletion does not
delete historical jobs because `prompt_schedule_id` is preserved as provenance
rather than a foreign key.

The migration follows the current Alembic head and covers both SQLite and
PostgreSQL without modifying existing queue rows.

## Cron evaluator

Add `app/services/cron_schedule.py` with small, independently testable pieces:

- `parse_cron(expression) -> CronExpression`
- `validate_timezone(name) -> ZoneInfo`
- `matches(local_datetime) -> bool`
- `next_fire_after(utc_datetime, timezone) -> utc_datetime`

Parsing must reject malformed fields, out-of-range values, zero/negative steps,
oversized expressions, and expressions with no future match in the supported
search horizon. Use a bounded field-jump search rather than checking every
minute, with a horizon long enough to include leap-day schedules. Never use
`eval` or shell out to system cron.

## Scheduling service

Add `app/services/prompt_scheduler.py`:

- `PromptScheduler.run_once(now=None)` finds enabled schedules whose
  `next_run_at` is due.
- Claim rows with `FOR UPDATE SKIP LOCKED` on PostgreSQL. The default SQLite
  deployment has one scheduler thread; the run provenance uniqueness constraint
  is the final duplicate guard.
- In one database transaction, insert the `prompt.inference` job, set
  `last_enqueued_at`, and advance `next_run_at`. A crash cannot advance a
  schedule without also persisting its job.
- Limit each scan to a bounded batch and one coalesced occurrence per schedule.
- `start_scheduler()` and `stop_scheduler()` manage a daemon thread with a stop
  event, mirroring the existing job worker lifecycle.
- Configuration:
  - `GEIST_PROMPT_SCHEDULER_ENABLED=true`
  - `GEIST_PROMPT_SCHEDULER_POLL_INTERVAL=15`
  - `GEIST_PROMPT_SCHEDULER_BATCH_SIZE=100`

An invalid persisted schedule is disabled with a logged error rather than
crashing the scheduler loop.

## Background inference path

### Shared inference service

Add `app/services/inference.py` to remove agent construction and completion
details from route/job code:

1. Load the default agent context through `agent_context_provider`.
2. Resolve the owning user's settings plus validated schedule overrides.
3. Create or retrieve the appropriate runtime.
4. Invoke `complete_text` with the standard system prompt and generation
   parameters.
5. Convert the result to `AgentCompletion` and return a JSON-safe dictionary
   containing `message`, `completion_id`, and `chat_id`.

Keep this service independent from `app.main` and accept an explicit `user_id`
for scheduled work. It reuses the same agent factory, default context, settings,
and completion contracts as interactive inference without forcing a risky route
refactor into this feature. Moving the legacy agent routes to mandatory
authentication remains a separate compatibility change.

Local model initialization is expensive. Preserve a bounded process cache keyed
by resolved runtime configuration, and serialize calls per cached local runtime
with a lock. Do not place API keys or secret values in cache keys, job payloads,
database rows, logs, or responses.

### Job handler and worker lanes

Register `@job_handler("prompt.inference")`. The handler validates the persisted
payload again, calls the shared inference service with `user_id`, and returns
the JSON-safe completion summary. Existing job retry/backoff records transient
provider or inference failures.

Extend claiming/worker construction with job-kind filters and run two lanes:

- the existing general worker handles tool and workflow jobs;
- one inference worker handles only `prompt.inference`.

This gives inference a dedicated background thread without allowing concurrent
access to local model runners or starving unrelated jobs. Preserve the current
public `start_worker`/`stop_worker` entry points behind a small supervisor so
callers and tests do not need broad changes.

## API

Add `app/schemas/prompt_schedule.py` and
`app/api/v1/endpoints/prompt_schedules.py`:

- `POST /api/v1/prompt-schedules` creates a schedule and computes its first
  `next_run_at`.
- `GET /api/v1/prompt-schedules` lists only the current user's schedules.
- `GET /api/v1/prompt-schedules/{id}` returns one owned schedule.
- `PATCH /api/v1/prompt-schedules/{id}` updates fields; cron/time-zone/enabled
  changes recompute `next_run_at` atomically.
- `DELETE /api/v1/prompt-schedules/{id}` deletes the schedule but preserves its
  historical jobs.
- `POST /api/v1/prompt-schedules/{id}/run` queues an immediate background run
  and returns `202 Accepted` with the `job_id`.
- `GET /api/v1/prompt-schedules/{id}/runs` lists owned job history for that
  schedule.

Register the router in `create_app`. Keep generic arbitrary job enqueueing
unexposed.

Update job access so scheduled jobs are visible only to the user recorded in
their `prompt_schedule_run`. Existing ownerless tool/workflow job behavior is
unchanged until ownership is added to those features.

## Chat tool

Add `CronScheduleAdapter`, discovered through the existing adapter registry,
with a `create_prompt_schedule` action. It accepts a name, prompt, cron
expression, time zone, and optional agent type, delegates validation and
persistence to the same schedule service as the HTTP API, and returns the
schedule ID plus next run time. The adapter binds schedules to Geist's current
default user; user IDs are never model-supplied tool arguments.

## Frontend tab

After the backend contract is established:

- Add a `Schedules` route and sidebar entry.
- Provide create/edit fields for name, prompt, five-field cron, time zone, and
  default/local/online runtime selection. Keep detailed model/generation
  overrides available through the API for the initial delivery.
- Show a human-readable next-run time returned by the backend rather than
  duplicating cron evaluation in TypeScript.
- Include enable/disable, run-now, and delete actions with loading/error states.

No frontend package is required.

## Failure and lifecycle behavior

- Creating or editing an invalid cron/time zone returns 422 before persistence.
- Editing a schedule never mutates already queued job payloads.
- Disabling or deleting a schedule prevents future enqueueing; already queued
  jobs remain durable and execute unless cancellation is added later.
- Handler failures use the existing bounded exponential retry policy.
- On graceful shutdown, stop the scheduler first so no new work is added, then
  stop worker threads.
- The existing queue can leave a job in `running` if the process dies
  mid-handler. A distributed lease/heartbeat recovery protocol remains a queue
  follow-up rather than requiring an incompatible job-table alteration here.

## Implementation sequence

1. Add cron parser/evaluator and exhaustive unit tests.
2. Add `prompt_schedule` and `prompt_schedule_run` plus their Alembic migration,
   database helpers, and tests.
3. Add the shared inference service and mocked local/online tests. Verify
   user-specific settings, JSON-safe results, chat persistence, and runtime
   locking.
4. Register `prompt.inference` and split the worker into general/inference lanes.
5. Add the scheduler's atomic scan/enqueue/advance transaction and lifecycle
   hooks.
6. Add current-user CRUD, run-now, run-history, and owner-scoped job APIs.
7. Add the chat scheduling adapter and its schema/dispatch tests.
8. Add the Schedules UI and frontend tests.
9. Run the repository-required verification loop before push.

## Tests

### Cron unit tests

- Wildcard, list, range, and step parsing.
- Field bounds, Sunday `0`/`7`, and day-of-month/day-of-week OR semantics.
- Invalid syntax and impossible schedules.
- Time-zone conversion, spring-forward skip, and fall-back single execution.

### Database and scheduler tests

- CRUD and per-user isolation.
- Correct first/next occurrence calculation.
- Disabled schedules are ignored.
- Due scan atomically inserts a job and advances the schedule.
- Two scans cannot create two jobs for the same occurrence.
- Downtime coalesces missed runs to one job.
- Schedule edits affect future jobs only.

### Worker and inference tests

- Scheduled prompt jobs run on the inference lane, not the general lane.
- General jobs continue while inference is blocked in a test double.
- Handler resolves the owning user's defaults and applies overrides.
- Successful results contain response text and `chat_id`.
- Failures retry and become terminal after `max_attempts`.
- Local runtime calls are serialized and the loaded model is reused.
- Startup/shutdown is idempotent and does not leak threads.

### API tests

- CRUD validation and current-user/404 ownership behavior.
- Run-now returns 202 and a pollable owned job.
- Run history cannot expose another user's prompt, payload, or response.
- Job list/detail endpoints enforce owner scoping.

### Tool and frontend tests

- The scheduling adapter is discovered and exposes only validated parameters.
- A chat tool call creates a schedule for the default user and cannot choose a
  different owner.
- The Schedules tab lists tasks and supports create, enable/disable, run-now,
  edit, and delete states.

## Verification

1. Run Ruff and MyPy over changed backend files.
2. Run focused cron, database, scheduler, job-worker, inference, and API tests.
3. Run affected agent and core-contract tests because the shared inference and
   worker contracts change.
4. Run backend tests in Docker as required by `AGENTS.md`.
5. Start with `docker compose up -d`, inspect backend logs, create a schedule a
   minute ahead, confirm the job completes, and curl `http://localhost:3000`.
6. Because local inference/runtime behavior is touched, use the Geist test-loop
   skill before pushing and exercise native `make run MLX_BACKEND=1` when the
   environment can load the configured model.

## Explicit non-goals for the first implementation

- Second-level cron schedules.
- Distributed scheduling across multiple SQLite application processes.
- Multiple application processes sharing one job queue (which requires job
  leases and heartbeats).
- Parallel local-model inference.
- Workflow scheduling (the provenance design can support it later).
- Job cancellation, notifications, webhooks, or output delivery beyond job
  status and chat history.
- Arbitrary shell commands or user-supplied Python execution.
