# Chat Stream Recovery and Browser E2E

## Goal

Prevent chat from remaining in `connecting` when agent or model initialization fails, keep MLX selection explicit to native Apple Silicon runtimes, and cover successful and failed streaming flows in a real browser.

## Implementation

1. Wrap the complete SSE generator boundary so backend exceptions are logged and converted into terminal `error` and `done` events after headers have been sent.
2. Allow `GEIST_LOCAL_RUNNER` to explicitly select a local runner and configure Linux Docker to use `transformers` instead of the MLX-only default.
3. Add focused backend tests for pre-stream initialization failures and runner override validation.
4. Add Playwright browser tests against the real React app and FastAPI stream route using a deterministic test-only agent server.
5. Run the browser suite in CI with the same lockfile-only dependency installation policy as the existing frontend job.

## Validation

- Ruff, formatting, and mypy for changed Python.
- Focused backend stream and factory tests.
- Frontend Jest tests and production build.
- Playwright browser tests for a successful chat response and a backend initialization failure that must leave `connecting`.
- Docker startup, logs, and HTTP checks.
- Native MLX chat using copied `llama_3_1` weights.
- Python and frontend dependency audits after lockfile changes.

## Runtime Handoff

Copy the Desktop weights into `app/model_weights/llama_3_1`, launch current main natively with `MLX_BACKEND=1`, and leave the UI open for manual retesting.
