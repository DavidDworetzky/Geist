# Implementation Plan: Online and Offline Image Generation as Async Tool Calls

## Overview

Image generation is slow, so it must never run inline in an agent tick or a
request thread. This plan (reworked after the database-backed job queue of
plan 152 landed) delivers image generation as the first **asynchronous tool
call**: the agent invokes the tool, immediately receives a job handle, and
checks for completion on a later tick.

## What exists

- The durable job queue (`app/models/database/job.py`,
  `app/services/job_queue.py`): enqueue/claim/retry semantics, a worker
  thread started with the app, and read-only `/api/v1/jobs` status endpoints.
- Structured tool calling (`agents/tool_calling.py`): schema-validated
  dispatch of adapter actions with function-log journaling.
- The adapter registry (`adapters/adapter_registry.py`): auto-discovers
  `BaseAdapter` subclasses and initializes them with environment config.

## Architecture

### 1) Async tool-call abstraction

- `adapters/async_tool.py` — `@async_tool` marks a slow adapter action for
  queued execution; `is_async_tool` checks the marker.
- `ToolDispatcher.dispatch` (agents/tool_calling.py): after schema
  validation, actions marked `@async_tool` are enqueued as a `tool.call` job
  instead of executed inline. The dispatch returns immediately with
  `{"async": true, "job_id": N, "status": "queued", "check_with":
  "JobStatusAdapter__check_async_tool"}`.
- `tool.call` job handler (app/services/job_queue.py): re-initializes the
  adapter from the environment inside the worker and executes the action.
  Only public actions actually marked `@async_tool` are runnable, and the
  payload is only ever produced by the dispatcher after validation.
- `adapters/job_status_adapter.py` — `JobStatusAdapter.check_async_tool(job_id)`
  is the agent-facing completion check: a normal (synchronous) tool the model
  polls until `done` is true, then reads `result` or `error`. Auto-discovered
  like every adapter, so it needs no special wiring.

This abstraction is generic: any future slow tool (video, long scrapes,
batch embedding) becomes async by adding one decorator.

### 2) Image generation adapters (split by provider and mode)

- `adapters/tool_modes.py` — `@online_tool` / `@offline_tool` decorators
  declare an action's execution mode as metadata (read with `tool_mode()`);
  they compose with `@async_tool` and are reusable on any adapter action.
- `adapters/image_gen_base.py` — `BaseImageGenAdapter` holds the shared run
  template (`_generate(prompt) -> list[bytes]` -> persist artifacts ->
  result payload) plus output configuration (`IMAGE_GEN_OUTPUT_DIR`,
  `IMAGE_GEN_TIMEOUT_SECONDS`). Abstract; excluded from adapter discovery.
- `adapters/gemini_image_adapter.py` — `GeminiImageAdapter.generate_image`,
  decorated `@async_tool @online_tool`: Gemini image generation API via
  httpx (already a dependency). Config: `GEMINI_API_KEY` (also injected via
  `app/environment.py`), `GEMINI_BASE_URL`, `GEMINI_IMAGE_MODEL`.
- `adapters/flux_image_adapter.py` — `FluxImageAdapter.generate_image`,
  decorated `@async_tool @offline_tool`: local generation extension point
  rather than bundled diffusion dependencies. `IMAGE_GEN_OFFLINE_BACKEND`
  names a module exposing `generate(prompt) -> list[bytes]` (e.g. a
  FLUX.1-schnell shim); without it, it fails with a clear message. This
  keeps the default install lean per repo preference.
- Artifacts are written under `output/images/<date>-<id>/image_N.png` and
  their paths returned in the job result.

### 3) Agent flow

1. Model calls `ImageGenAdapter__generate_image` with a prompt.
2. Dispatcher enqueues `tool.call`, returns the job handle in the tool
   result; the handle is journaled to the function log like any result.
3. On later ticks the model calls
   `JobStatusAdapter__check_async_tool(job_id=N)` until `done`, then uses
   `result.file_paths`.

No new tables, endpoints, or workers: job state is visible through the
existing `/api/v1/jobs` endpoints and executed by the existing worker.

## Superseded from the original plan

- Bespoke `app/services/image_queue.py` (asyncio.Queue) → the job queue.
- `ImageJob`/`ImageArtifact` tables and migrations → `job` rows + files on
  disk (structured artifact rows can come later if querying demands it).
- Dedicated `/image/gen` endpoints → agent tool calls + `/api/v1/jobs`.

## Tests

- `tests/agents/test_async_tool_calls.py`: async dispatch returns a queued
  job handle and enqueues exactly one `tool.call` job; sync tools still
  execute inline; the `tool.call` handler runs the real action through the
  worker; `JobStatusAdapter` reports queued → succeeded/failed transitions;
  non-`@async_tool` actions are refused by the handler.
- `tests/adapters/test_image_gen_adapters.py`: mode decorators and async
  markers on both adapters; discovery includes both concrete adapters and
  excludes the abstract base; Gemini calls carry the prompt and API key and
  write returned images to disk (httpx mocked); FLUX runs through a
  configured backend module and errors clearly without one; missing API
  key / empty outputs produce clear errors.
