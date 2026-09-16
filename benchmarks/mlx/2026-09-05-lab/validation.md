# MLX laboratory validation — 2026-09-05

Checkout: `/private/tmp/geist-mlx-100tps.2ATYwp`, branch `codex/mlx-100tps-lab`,
starting commit `eef0ae3a830dd7179c4f873cfb3ea25e66fa485b`.
The original user worktree and PR #356 were left unchanged.

No dependencies were installed or changed. Commands used the existing native
environment and the existing `geist-goal-decomposition-backend:latest` image.
Native Metal commands ran outside the restricted sandbox.

## Native MLX: PASS

Working directory: the checkout above.

```sh
/Users/daviddworetzky/Documents/repos/Geist-wt/geist-6/.venv/bin/python -m pytest \
  tests/agents/test_mlx_dflash.py \
  tests/agents/test_ngram_draft.py \
  tests/agents/test_speculation_policy.py -q --durations=5
```

Final result: **151 passed, 3 warnings in 6.80s**. Warnings concern existing
Pydantic class-based configuration deprecation. Coverage includes hybrid cache
rollback, target/drafter reuse, adaptive transitions, output limits, cancellation,
EOS, sampling contracts, ngram continuity, compiled recurrence, expanded weights,
and experimental matmul numerical checks.

Real target/drafter performance runs are separate evidence; see the JSON files
and the feature log. Small-model correctness tests do not certify the accuracy
of all real-checkpoint outputs.

## Docker focused contracts: PASS

```sh
docker run --rm --name geist-mlx-lab-contracts --platform linux/amd64 \
  --memory 4g --entrypoint python \
  -v /private/tmp/geist-mlx-100tps.2ATYwp:/work:ro \
  -w /work -e PYTHONPATH=/work \
  geist-goal-decomposition-backend:latest -m pytest \
  tests/agents/test_dflash_artifact.py \
  tests/agents/test_mlx_llama_runner.py \
  tests/agents/test_model_catalog.py \
  tests/agents/test_new_architecture.py \
  tests/agents/test_generic_completion.py \
  tests/agents/test_native_tool_turns.py \
  tests/agents/test_ngram_draft.py \
  tests/agents/test_speculation_policy.py \
  tests/services/test_local_models.py \
  tests/services/test_user_settings_artifacts.py \
  tests/services/test_user_settings_service.py -q -p no:cacheprovider
```

Final result: **227 passed, 3 warnings in 0.97s**. These are Linux contract/service
tests, not proof of Apple GPU execution. The container exited and was removed.

## Static checks: PASS

The following changed Python paths were passed explicitly to Ruff:

```text
agents/architectures/llama/dflash_backend.py
agents/architectures/llama/mlx_lm_backend.py
agents/architectures/llama/ngram_draft.py
agents/architectures/llama/qwen_compiled_verifier.py
agents/architectures/llama/qwen_expanded_mlp.py
agents/architectures/llama/qwen_kernel_lab.py
agents/architectures/llama/speculation_policy.py
scripts/benchmark_mlx_dflash.py
scripts/benchmark_mlx_dispatch.py
scripts/benchmark_mlx_devices.py
scripts/profile_mlx_verifier.py
tests/agents/test_mlx_dflash.py
tests/agents/test_ngram_draft.py
tests/agents/test_speculation_policy.py
```

- Existing-venv `ruff check`: all checks passed.
- Existing-venv `ruff format --check`: 14 files already formatted.
- Existing-venv `mypy --config-file=pyproject.toml` on the first 11 source
  paths: no issues found. Existing llama-package error suppression applies;
  this is not a strict type-check guarantee for that package.
- `git diff --check`: passed.
- No full all-repository pre-commit or frontend test run was substituted for
  targeted inference verification. No dependency changes require a new audit.

## Isolated Docker API startup: PASS

```sh
docker run --rm -d --name geist-mlx-lab-smoke --platform linux/amd64 \
  --memory 4g --entrypoint python -p 127.0.0.1:5586:5001 \
  -v /private/tmp/geist-mlx-100tps.2ATYwp:/opt/geist:ro -w /opt/geist \
  -e PYTHONPATH=/opt/geist \
  -e SQLALCHEMY_DATABASE_URL=sqlite:////tmp/geist-mlx-lab.sqlite3 \
  -e GEIST_DATABASE_PROVIDER=sqlite -e GEIST_MARKDOWN_ROOT=/tmp/geist-markdown \
  -e XDG_DATA_HOME=/tmp/geist-data \
  geist-goal-decomposition-backend:latest -c \
  'import dotenv; dotenv.load_dotenv=lambda *a,**k:False; from app.database_upgrade import upgrade_database; upgrade_database(); import uvicorn; uvicorn.run("app.main:app",host="0.0.0.0",port=5001,log_level="info")'

docker logs --tail 80 geist-mlx-lab-smoke
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:5586/docs
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:5586/openapi.json
docker stop geist-mlx-lab-smoke
```

Both endpoints returned **HTTP 200**. Startup completed; observed logs contained
no errors or tracebacks. Only this disposable test container and its ephemeral
database were removed by the stop/`--rm` lifecycle; there was no user data in it.

## Isolated native API startup: PASS

Created a scratch directory with:
`mktemp -d /private/tmp/geist-mlx-smoke.XXXXXX`, which returned the directory
used below. For reproduction, create a fresh directory and substitute its path.

```sh
SQLALCHEMY_DATABASE_URL=sqlite:////private/tmp/geist-mlx-smoke.0oSSGB/native.sqlite3 \
GEIST_DATABASE_PROVIDER=sqlite \
GEIST_MARKDOWN_ROOT=/private/tmp/geist-mlx-smoke.0oSSGB/markdown \
XDG_DATA_HOME=/private/tmp/geist-mlx-smoke.0oSSGB/data \
MLX_BACKEND=1 \
/Users/daviddworetzky/Documents/repos/Geist-wt/geist-6/.venv/bin/python -c \
'import dotenv; dotenv.load_dotenv=lambda *a,**k:False; from app.database_upgrade import upgrade_database; upgrade_database(); import uvicorn; uvicorn.run("app.main:app",host="127.0.0.1",port=5587,log_level="info")'

curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:5587/docs
curl -fsS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:5587/openapi.json
```

Both endpoints returned **HTTP 200**. The terminal stayed open throughout the
checks, then received Ctrl-C; logs confirmed application shutdown completed.
This only validates native API startup/schema, not model-backed API chat.
Actual GPU inference is independently covered by the native suite and benchmarks.

For both smoke processes, dotenv loading was disabled in that process only,
before importing the application. No local secret files were read or modified.
Temporary native smoke data remain in the scratch directory.

## Explicitly not run

- `docker compose up -d --build` and `curl http://localhost:3000` for this
  checkout: the standard ports were owned by another worktree's app.
- Full `make run MLX_BACKEND=1`: an isolated existing-venv native API was used
  to avoid displacing that app or bootstrapping unapproved dependencies.
- Browser chat and reversible settings save/reload: no lab frontend was started.
  The already-running UI was not counted as evidence for this branch.

No full UI or packaged-default qualification is claimed. At the end of the
initial research pass the new flags remained opt-in; the subsequent promotion
and pre-PR checks are recorded below.

## User-requested branch-default promotion: PASS with UI limits

The application now constructs adaptive DFlash automatically for the supported
Qwen model when its drafter exists. Shape-qualified Metal routing is unchanged.
Copying, expanded weights and compiled recurrence remain benchmark-only.

Reran the native command above without `--durations=5`: **151 passed, 3 warnings
in 6.28s**. Reran the exact Docker focused-contract command above after adding
the no-opt-in default/disabled-path regression cases: **229 passed, 3 warnings
in 1.17s**. Warning class is unchanged. Targeted lint/format and staged
repository pre-commit checks use existing environments; no dependency install
is authorized or needed. Mypy retains the repository's existing exclusions.

The real-model activation check used `MLXLMBackend` directly, not a manually
constructed experimental decoder. It disabled dotenv loading in that process,
set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, removed only process-local
`GEIST_MLX_DFLASH` and `GEIST_MLX_DFLASH_DIR` overrides, and loaded the installed
target snapshot. Constructor settings were `max_new_tokens=128`, `temperature=0`,
`top_p=1`, `model_id="Qwen/Qwen3.8-27B"`. It called `stream_text("", prompt)` on
the extended-suite essay, then set `temperature=0.7` and streamed a request to
merge two sorted Python lists. The results are in
[branch-default-smoke.json](branch-default-smoke.json).

Both emitted 128 tokens and reported `implementation=mlx_dflash`, `adaptive=true`,
`adaptation_policy=cumulative-eight-v2`. The essay switched at token 21;
sampled code stayed speculative. Assertions verified no copy window, the
uncompiled target class, and ordinary small-row wrapper settings with no
packed weights/half operands/direct fragments. All 395 installed wrappers
qualified in this run. These short outputs are not task-quality evaluations.

Reran the isolated API commands above against the final code, using Docker
container name `geist-mlx-default-smoke`, SQLite path
`/tmp/geist-mlx-default.sqlite3`, and native scratch directory
`/private/tmp/geist-mlx-default-smoke.eDGI2l`. Ports remained 5586 and 5587.
All four `/docs` and `/openapi.json` curl requests returned HTTP 200; observed
logs contained no errors or tracebacks. The native process shut down cleanly
on Ctrl-C and the disposable Docker container was stopped/auto-removed. Native
temporary data remain in its scratch directory; no user services were stopped.

The standard ports still belong to another worktree. Full Compose, `make run`
and browser chat/settings are therefore still not claimed; the PR explicitly
reports these limitations for the user's hands-on testing. The original
worktree and PR #356 remain unchanged; the new PR is stacked on #356.
