---
name: geist-test-loop
description: This skill should be used when Codex is preparing to push, publish, open a PR, or verify Geist runtime behavior through Docker, native startup, browser UI, chat, or settings smoke tests.
---

# Geist Test Loop

Use this skill to produce evidence that a Geist change works before pushing or publishing it. Treat this as an AI operator workflow, not a Git-enforced gate.

## Core Rules

1. Inspect `git status --short` before testing. Preserve unrelated user changes and report when uncommitted files are outside the current task.
2. Prefer the smallest meaningful test set first, then expand to runtime smoke checks when preparing to push or publish.
3. Keep Docker runtime validation separate from native `MLX_BACKEND=1` validation. A pass in one mode does not prove the other mode.
4. Do not print secrets. Confirm only presence, absence, or variable names when environment configuration matters.
5. Report exact commands run, pass/fail status, and any blocked checks in the final response.

## Fast Checks

Run relevant fast checks before runtime smoke tests:

```bash
pre-commit run --all-files
```

When pre-commit is unavailable or too broad for the task, run targeted checks:

```bash
ruff check .
ruff format --check .
mypy --config-file=pyproject.toml <changed-python-files>
cd client/geist && CI=true npm test -- --watchAll=false --passWithNoTests
```

For backend behavior, prefer the containerized test command from `AGENTS.md` when Docker is available:

```bash
docker compose up -d db backend
docker exec backend /bin/bash -lc 'cd /opt/geist && PYTHONPATH=/opt/geist pytest'
```

## Docker Smoke Loop

Run this loop before pushing or publishing user-facing changes when feasible:

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=120 backend
docker compose logs --tail=120 frontend
curl -fsS http://localhost:3000
```

Treat backend/frontend startup errors, repeated tracebacks, failed health checks, or a failed `curl` as blocking unless there is a clear unrelated local-environment cause.

## Browser UI Smoke

Use the browser-control skill or available browser tooling to verify the running app at:

```text
http://localhost:3000
```

Check for:

- The app renders a nonblank UI.
- The browser console has no obvious runtime errors from the current change.
- A basic chat flow can be reached and exercised without crashing.
- Settings can be opened, a reversible default/preference can be changed, saved, reloaded, and restored.

Use existing UI labels and routes discovered from the current app. If the chat flow requires a model, token, or external service that is unavailable, verify the UI path up to that dependency and report the blocker explicitly.

## Native MLX Smoke Loop

Run the native loop when the change touches local agents, model defaults, MLX behavior, native bootstrap, or when explicitly requested:

```bash
make run MLX_BACKEND=1
```

In native mode:

- Keep the terminal session open while testing.
- Verify the frontend/backend route that the native process exposes.
- Exercise the same basic chat and settings paths if the app reaches the UI.
- Stop the native process cleanly after testing.

If Conda, MLX, model weights, GPU capability, or local tokens are missing, report native validation as blocked rather than substituting Docker results.

## Pre-Push Evidence

Before pushing, summarize:

- Git branch and dirty-worktree state.
- Fast checks run.
- Docker startup/log/curl result.
- Browser chat/settings result.
- Native `MLX_BACKEND=1` result, or why it was not applicable.
- Any skipped checks and the reason.
