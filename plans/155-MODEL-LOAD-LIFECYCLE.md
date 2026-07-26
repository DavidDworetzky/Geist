# Model Load Lifecycle Indicator

## Problem

The chat UI labels a first local-model request as `connecting` while the backend
downloads and loads model weights. Large models can spend minutes in that state
without explaining what is happening.

## Contract

- Track local model lifecycle state in the backend as `unloaded`, `loading`,
  `ready`, or `failed`.
- Expose `GET /api/v1/models/status/{model_id}` so clients can poll a selected
  model without starting another load.
- Have local runners mark the model `loading` before expensive initialization,
  `ready` after the runner is usable, and `failed` when initialization raises.
- Keep status state process-local because loaded runner instances are also
  process-local.

## Frontend

- Poll the status endpoint while a local chat turn is waiting for its first
  stream event and the model is not ready.
- Replace the ambiguous `connecting` label with a model-specific loading message
  only while the backend reports `loading` or `failed`; an `unloaded` response
  does not claim that loading has begun.
- Stop polling once the model is ready, failed, the stream starts, or the turn is
  cancelled.
- Ignore lifecycle records older than the active turn and back off transient
  polling failures while stopping immediately when the model endpoint is absent.

## Tests

- Unit-test lifecycle transitions and defensive copies in the backend registry.
- Route-test unloaded and active model status responses.
- Verify runner success/failure updates the lifecycle.
- Test frontend polling, loading-state rendering, and polling termination.
- Isolate the process-local registry between backend tests and cover unknown
  model responses.
- Rebuild and smoke-test the Docker backend/frontend and inspect the chat UI in a
  browser.
