# Geist QA Runtime Matrix

Use this reference after ranking the PR's affected pathways. A lane is required only when the diff can change behavior in that lane, except Docker-default, which is the baseline for any user-facing or backend runtime change.

## Lane selection

| Lane | Require when the diff affects | Real-model proof |
|---|---|---|
| Docker default | Frontend, API, service, database, agent orchestration, runtime configuration, or user-visible behavior | Send one real prompt when inference is affected; otherwise prove the changed behavior |
| Native MLX | MLX runners, local agents, completion/message contracts, model selection/catalog, streaming, tools, or shared inference services | Confirm `mlx_llama` and `Qwen/Qwen3-4B`, then receive a real response |
| llama.cpp | `llama_server*`, GGUF/model management, native sidecar, local-agent selection, completion/message contracts, streaming, tools, or shared inference services | Confirm `llama_server`, Qwen3 GGUF, and a real response |
| Transformers | Transformers runners, Docker inference, local-agent selection, model catalog, completion/message contracts, streaming, tools, or shared inference services | Confirm `transformers` and `Qwen/Qwen3-4B`, then receive a real response |

When a shared core inference contract changes, require all four lanes. When a frontend-only change does not touch inference, use Docker-default plus the focused browser pathway and mark the other lanes not applicable in the traceability report.

## Common preflight

Before starting a lane:

1. Verify required executables and already-installed locked dependencies without installing anything.
2. Verify the required local artifact through the Models API/UI or the model manager's public status surface. Do not inspect local secret files.
3. Verify ports before binding them. Prefer repository defaults; report alternate ports exactly when conflicts require them.
4. Use temporary SQLite data under `/private/tmp` when the lane permits an override and testing must not disturb persistent user state.
5. Record runner identity, model identity, health response, focused-test output, browser result, console errors, and relevant log lines.

Missing dependencies, model weights, platform support, screen-recording permission, or required services make an applicable lane `BLOCKED`. Do not download a multi-gigabyte model or install an optional extra during QA without explicit approval.

## Docker default

The Compose stack defaults to `GEIST_LOCAL_RUNNER=transformers` and exposes the frontend on port `3000` and backend on `5001`.

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=120 backend
docker compose logs --tail=120 frontend
curl -fsS http://127.0.0.1:5001/health/ready
curl -fsS http://127.0.0.1:3000
```

For focused backend tests in the running backend container:

```bash
docker compose exec -T backend /bin/bash -lc \
  'cd /opt/geist && PYTHONPATH=/opt/geist pytest -q <focused-tests>'
```

Confirm the actual runner from backend logs or an application status response. If inference is affected, select an already-installed Qwen 3 model and send a real prompt.

## Native MLX

Use only on supported Apple silicon with the locked MLX extra already present and Qwen 3 weights already installed.

```bash
make services
GEIST_LOCAL_RUNNER=mlx_llama make run MLX_BACKEND=1
```

Select `Qwen/Qwen3-4B`, send the deterministic prompt, and confirm the backend reports `mlx_llama`. Keep the native process open during browser/API validation and stop it cleanly afterward.

If `make run MLX_BACKEND=1` fails specifically because reload/watch behavior is not permitted, a direct non-reloading `uv run uvicorn app.main:app --host 127.0.0.1 --port 5001` launch may be used with the same explicit runner configuration. Record the fallback.

## llama.cpp

Use the managed runner only on a supported platform with an existing `llama-server` binary and installed Qwen3 GGUF artifact.

```bash
GEIST_LOCAL_RUNNER=llama_server make run
```

Select the managed Qwen3 4B Q4_K_M artifact, send the deterministic prompt, and confirm runner, model, and managed child-process readiness from application status/logs.

The current managed artifact policy supports llama.cpp on Windows x64 and Linux x64. On an unsupported host, mark an applicable real-runtime check `BLOCKED`; focused process/contract tests may add evidence but cannot replace it.

## Transformers

Use an existing locked Transformers environment and existing Qwen 3 weights. Docker is the normal portable lane; a native run is acceptable when it better isolates runner behavior.

```bash
GEIST_LOCAL_RUNNER=transformers docker compose up -d --build
```

Or, when the native locked extra is already installed:

```bash
GEIST_LOCAL_RUNNER=transformers make run
```

Select `Qwen/Qwen3-4B`, send the deterministic prompt, and confirm `transformers` from logs or a status response. A mocked runner or the E2E fixture server is useful focused evidence but is not real-model proof.

## Browser pathway

Use the installed browser-control skill against the lane's frontend URL. Keep the interactive pathway short:

1. Open the changed entry point.
2. Exercise the highest-ranked happy path.
3. Exercise one cheap negative or recovery path when risk justifies it.
4. Confirm visible result, persisted state when applicable, and absence of new browser-console errors.
5. Correlate the action with backend logs or an API response in the terminal.
6. Restore reversible state.

Do not spend recording time waiting for builds or model downloads. Include failures exactly as observed.
