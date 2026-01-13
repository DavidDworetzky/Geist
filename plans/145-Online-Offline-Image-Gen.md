# Implementation Plan: Online and Offline Image Generation Adapters

## Overview
Add two image-generation adapters and the supporting backend plumbing:
- Offline image generation using FLUX.1-schnell as the default local tool, executed via an async job queue to avoid blocking request threads.
- Online image generation using Gemini's default image generation API via an HTTP-backed adapter.

These adapters will follow the existing `adapters` pattern and integrate with services, storage, and API endpoints to create, track, and retrieve generated images.

## Goals & Non-Goals
- Goals
  - Provide a consistent adapter interface for image generation (enumerable actions, adapter registry integration).
  - Implement an offline queue-backed worker that runs FLUX.1-schnell locally and stores results.
  - Implement an online adapter calling Gemini image-gen with configurable API keys and endpoints.
  - Expose API endpoints to submit image jobs, query job status, and download artifacts.
  - Persist job metadata and outputs in the database and file storage.
  - Add tests covering adapters, queue behavior, and API flows.
- Non-Goals
  - Frontend UI for image generation (can be added later; ensure API is stable).
  - Shipping multiple offline backends beyond FLUX.1-schnell (provide extensible scaffolding; one default backend).
  - Advanced prompt schedulers or multi-image batch pipelines (out of scope for first pass).

## Current Architecture Assessment
- `adapters/` contains pluggable adapters with `BaseAdapter` and an `adapter_registry` pattern.
- `app/services/` includes utilities like `file_storage.py` that can store artifacts.
- `app/api/v1/endpoints/` exposes REST endpoints; new image-gen endpoints can be added here.
- Database models exist under `app/models/database/` and are managed by Alembic migrations.
- Logging utilities exist under `utils/logging.py`.

## Proposed Architecture
### 1) Image Generation Adapters
- Create `adapters/gemini_image_adapter.py` (online):
  - Actions: `generate_image(prompt: str, params: dict) -> str|List[str]` (returns artifact path(s) or job id based on mode), `enumerate_actions`.
  - Configuration: `GEMINI_API_KEY`, `GEMINI_IMAGE_ENDPOINT` or provider SDK base URL; timeouts and retries.
  - Responsibility: Synchronous call to provider. For long-running provider jobs, return a job id and poll (V1 will attempt synchronous single-image generation if supported).
- Create `adapters/flux_schnell_adapter.py` (offline):
  - Actions: `enqueue_image_job(prompt: str, params: dict) -> str` (returns `image_job_id`), `enumerate_actions`.
  - Responsibility: Submit tasks to an internal async queue; a background worker consumes jobs and writes artifacts.
  - Default backend: FLUX.1-schnell local runner. Prefer minimal external deps; use local Python execution where possible.

### 2) Offline Queue and Worker
- Service: `app/services/image_queue.py`
  - Uses `asyncio.Queue` for in-process job buffering and a background task running on app startup.
  - Job payload includes: `job_id`, `prompt`, `params`, `adapter_name`, `output_basename`, and optional `user_id`.
  - Worker resolves the offline backend (FLUX.1-schnell) and generates image(s); updates DB state and stores artifacts via `file_storage`.
- Service: `app/services/image_generation.py`
  - Abstraction to create jobs, dispatch to appropriate adapter (online/offline), and normalize results.
  - Encapsulates interaction with `file_storage` and DB.

### 3) FLUX.1-schnell Local Runner
- Module: `agents/architectures/flux/runner.py` (or `agents/architectures/flux_runner.py` if flatter)
  - API surface:
    - `load(model_path: str, device_config: dict) -> None`
    - `generate(prompt: str, params: dict) -> List[bytes] | List[ndarray] | List[Path]`
    - `cleanup() -> None`
  - Keep implementation minimal; prefer built-in libs and existing infra. If model weights are needed locally, wire into `scripts/copy_weights.py` and `app/model_weights/`.
  - If FLUX.1-schnell requires a specific runtime, add a thin shim that calls a CLI or Python module, capture outputs, and write images to disk.

### 4) Online Gemini Image Generation
- Adapter calls Gemini's image generation endpoint (Imagen/Gemini image-gen depending on availability) using `httpx` or `requests` (prefer minimal deps; `httpx` if already present, otherwise `requests`).
- Configuration via env: `GEMINI_API_KEY`, `GEMINI_REGION` (if needed), `GEMINI_IMAGE_MODEL` (e.g., default model name), `GEMINI_BASE_URL`.
- Response handling writes image bytes to storage and records a job row (synchronous path) or records a job for polling if the provider is async.

### 5) Data Model and Persistence
- New SQLAlchemy models under `app/models/database/`:
  - `ImageJob` table
    - `image_job_id` (PK), `prompt`, `params_json`, `adapter_name`, `status` (queued|running|succeeded|failed), `error_message`, `created_at`, `updated_at`.
  - `ImageArtifact` table
    - `image_artifact_id` (PK), `image_job_id` (FK), `file_path`, `format`, `width`, `height`, `metadata_json`, `created_at`.
- Alembic migrations under `migrations/versions/` to create these tables.
- Use existing `app/services/file_storage.py` to persist image files under `output/` (e.g., `output/images/{job_id}/image_*.png`).

### 6) API Endpoints
- Add FastAPI routes under `app/api/v1/endpoints/image_gen.py`:
  - `POST /image/gen` – body: `prompt`, `mode` (online|offline), `params` (optional), returns `job_id`.
  - `GET /image/jobs/{job_id}` – returns job status and artifact metadata.
  - `GET /image/jobs/{job_id}/artifacts/{artifact_id}` – streams image file.
- Pydantic schemas in `app/schemas/image.py`:
  - `ImageGenRequest`, `ImageJobStatus`, `ImageArtifactResponse`.

### 7) Adapter Registry Integration
- Ensure `adapters/adapter_registry.py` can initialize `GeminiImageAdapter` and `FluxSchnellAdapter`.
- Each adapter implements `enumerate_actions` appropriately and follows constructor arg filtering rules already present.

### 8) Configuration & Secrets
- Add env vars and load via existing settings modules (e.g., `app/environment.py`):
  - `GEMINI_API_KEY`, `GEMINI_BASE_URL`, `GEMINI_IMAGE_MODEL`.
  - `FLUX_MODEL_PATH` (if required), `FLUX_DEVICE`.
- Document these in `docs/agents.md` or a new `docs/image_gen.md`.

### 9) Logging & Observability
- Use `utils/logging.py` to create module-scoped loggers with `__name__`.
- Log job lifecycle transitions and durations.
- Include minimal metrics (e.g., counts per adapter, average latency) via simple counters in logs for now.

### 10) Backward Compatibility
- Entirely additive. New tables, endpoints, and adapters do not alter existing agent/completion flows.

## Implementation Plan
### Phase 1: Planning and Scaffolding
1. Create adapters: `adapters/gemini_image_adapter.py`, `adapters/flux_schnell_adapter.py` with `enumerate_actions` and stub methods.
2. Add `app/schemas/image.py` Pydantic models.
3. Add service scaffolding: `app/services/image_generation.py`, `app/services/image_queue.py`.

### Phase 2: Data Model and Migrations
1. Add `ImageJob` and `ImageArtifact` SQLAlchemy models under `app/models/database/` following repo conventions (inherit `Base`, add `__tablename__`, autoincrement PKs).
2. Create Alembic migration for both tables.

### Phase 3: Offline FLUX.1-schnell Backend
1. Implement `agents/architectures/flux/runner.py` with `load/generate/cleanup` stubs; wire to actual generation path.
2. Integrate runner with `FluxSchnellAdapter` and the async worker in `image_queue`.
3. Write images to `output/images/{job_id}/`; record artifacts.
4. If local weights are needed, update `scripts/copy_weights.py` to manage them under `app/model_weights/flux/`.

### Phase 4: Online Gemini Adapter
1. Implement HTTP calls to Gemini image generation using configured model and API key.
2. Handle response image bytes; persist via `file_storage`; record artifacts and success/failure.
3. Add simple retry/backoff for transient failures.

### Phase 5: API Endpoints
1. Implement endpoints in `app/api/v1/endpoints/image_gen.py` for submit/status/artifact-get.
2. Wire endpoints to `image_generation` service, which dispatches to online/offline adapters.
3. Update `app/api/utils.py` if shared helpers are needed.

### Phase 6: Startup & Worker Lifecycle
1. In `app/main.py`, create on-startup background task for `image_queue` worker; ensure graceful shutdown.
2. Add configuration toggles for enabling offline worker.

### Phase 7: Tests and Docs
1. Unit tests:
   - Adapters: action enumeration and basic flows.
   - Queue: enqueue, run, finalize job transitions; artifact creation.
   - API: submit/status/artifact endpoints (mock Gemini for online path).
2. Integration tests: end-to-end offline job run using a lightweight/dummy FLUX path if heavy model is unavailable in CI.
3. Documentation updates: environment variables, usage examples, and curl snippets.

## Testing & Verification
- Run `docker compose up -d`; verify no backend error logs.
- Offline flow: `curl -X POST http://localhost:3000/image/gen -H 'Content-Type: application/json' -d '{"prompt":"a cat on a bike","mode":"offline"}'` → poll status endpoint until `succeeded`; fetch artifact.
- Online flow: same as above with `mode":"online"` after setting `GEMINI_API_KEY`.

## Notes & Preferences Alignment
- Follow SQLAlchemy conventions from repo rules: import `Base, Session, SessionLocal` as specified; models include `__tablename__` and autoincrement integer PKs.
- Prefer minimal dependencies; implement queue with `asyncio` and built-ins.
- Use `__name__` for loggers.
- Use keyword arguments when calling functions.


