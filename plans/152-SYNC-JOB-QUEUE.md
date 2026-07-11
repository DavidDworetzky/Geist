# 152 - Sync Job Queue

## What exists

- No job queueing anywhere in the backend. `POST /api/v1/workflows/{id}/run`
  executes `WorkflowExecutor.execute_workflow` synchronously inside the request
  (blocking the event loop for long workflows). Agent ticks either run inline or
  as a fire-and-forget `subprocess.Popen('tick.py')` with no result channel.
- `WorkflowRun` rows journal execution status (running/completed/failed) but are
  not a queue: there is no queued state and nothing claims work later.
- The repo preference (claude.md) is minimal inline implementations over new
  dependencies; persistence is SQLAlchemy over SQLite (default) or PostgreSQL.

## What to add

A compact, durable, pure-Python job queue on the existing database — no broker,
no new dependencies.

### Data model (`app/models/database/job.py`)

`job` table: `job_id`, `kind` (handler key), `payload` (JSON text),
`status` (queued | running | succeeded | failed), `attempts`, `max_attempts`,
`run_after` (visibility time for delays/retry backoff), `result` (JSON text),
`error`, `created_at`, `updated_at`. Module-level helpers in the established
style: `enqueue_job`, `claim_next_job`, `mark_job_succeeded`, `mark_job_failed`
(requeues with exponential backoff until `max_attempts`, then fails),
`get_job`, `get_jobs`.

Claiming is one transaction: select the oldest visible queued row, flip it to
running, increment attempts. On PostgreSQL the select adds
`FOR UPDATE SKIP LOCKED` so multiple workers never double-claim; SQLite is a
single-writer database so the plain transaction is already safe.

### Service (`app/services/job_queue.py`)

- Handler registry: `register_job_handler(kind, fn)` / `@job_handler(kind)`.
  Handlers take the payload dict and return a JSON-serializable result.
- `enqueue(kind, payload, max_attempts, delay_seconds)` — insert one row.
- `JobWorker`: single background thread; `run_once()` claims and dispatches one
  job (unknown kind or handler exception -> `mark_job_failed`), loop polls at a
  configurable interval. `start_worker()`/`stop_worker()` manage a process
  singleton, gated by `GEIST_JOB_WORKER_ENABLED` (default on) and
  `GEIST_JOB_POLL_INTERVAL`.
- Built-in `workflow.run` handler that loads the workflow and calls the
  existing `WorkflowExecutor`, so workflow runs can be queued.

### API (`app/api/v1/endpoints/jobs.py` + `app/schemas/job.py`)

- `POST /api/v1/jobs` — enqueue (400 on unregistered kind).
- `GET /api/v1/jobs/{job_id}` — status/result polling.
- `GET /api/v1/jobs` — list, optional status filter.
- `POST /api/v1/workflows/{id}/run?background=true` — enqueue instead of
  executing inline; response carries `job_id` with `status: queued`.
  Default behavior is unchanged (synchronous).

### Wiring

- Router registered in `create_app`; worker started/stopped via FastAPI
  startup/shutdown events.
- Alembic migration `add job table` on head `b8c4d2e6f0a3`; model imported in
  `app/models/database/__init__.py` so `initdb.py` creates it.

### Tests

- `tests/database/test_job_queue.py` — enqueue/claim FIFO ordering, delayed
  visibility, succeed/fail transitions, retry backoff to terminal failure.
- `tests/services/test_job_worker.py` — worker dispatch to registered handlers,
  result persistence, handler-exception retry, unknown-kind failure, and the
  `workflow.run` handler end-to-end on SQLite.
