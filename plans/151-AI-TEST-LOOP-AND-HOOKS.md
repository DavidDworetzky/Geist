# AI Test Loop and Hook Plan

## Goal

Make Codex default to a real Geist smoke-test loop before pushing or publishing changes, while keeping Git hooks focused on fast linting and basic security checks.

## Scope

- Add a repo-local skill that tells Codex how to run Docker, native, UI, chat, and settings smoke checks before push/PR work.
- Keep the AI smoke loop advisory and evidence-driven, not enforced by Git.
- Extend pre-commit coverage with deterministic local checks that do not require standing up the whole application.

## Implementation

1. Add `.agents/skills/geist-test-loop/SKILL.md`.
   - Trigger when preparing to push, publish, create a PR, or verify Geist runtime behavior.
   - Prefer changed-file targeted tests first.
   - Run Docker smoke checks and browser UI checks before push when feasible.
   - Run native `MLX_BACKEND=1` checks when the change touches native/local model behavior or when explicitly requested.
   - Report exact commands, pass/fail state, and blockers.

2. Add a staged secret scanner script.
   - Scan staged file contents through Git.
   - Detect narrow high-signal token/private-key patterns.
   - Avoid printing secret values.

3. Add frontend lint wiring.
   - Add an npm `lint` script for `client/geist/src`.
   - Add a pre-commit local hook that runs frontend lint only when frontend source files are staged.

4. Update `.pre-commit-config.yaml`.
   - Keep existing Ruff, Ruff format, mypy, and generic safety hooks.
   - Add repo-local hooks for staged secret scanning and frontend lint.

## Verification

- Compile the staged secret scanner.
- Run the staged secret scanner.
- Validate JSON syntax for `client/geist/package.json`.
- Do not run network-dependent pre-commit installation or full Docker smoke unless requested.
