# Selected Local Artifact Routing Plan

## Base

- Base branch: `claude/agent-plugin-standard-geist-hhd03b` (PR #312)
- Head branch: `codex/local-native-tool-calling`

## Goal

Fix the display/runtime mismatch where an explicitly selected installed artifact
reports its persisted backend but `AgentFactory` re-infers a different runner from
the model ID and host platform.

## Implementation

1. Inspect `AgentFactory`, workspace settings, local artifact resolution, and
   runtime-info contracts to establish intended explicit-runner precedence.
2. Make a selected installed artifact's backend authoritative when no explicit
   runner override is supplied.
3. Preserve explicit runner overrides, and reject incompatible explicit
   artifact/runner combinations instead of silently loading the wrong runtime.
4. Add focused tests for Qwen GGUF routing on macOS, MLX artifact routing,
   settings/runtime reporting, explicit precedence, and mismatch failures.

## Verification

1. Run focused AgentFactory, settings, and runtime-info tests.
2. Run relevant lint/type checks and review the diff.
3. Run the Docker build/start/log/curl smoke loop.
4. Run native `make run MLX_BACKEND=1` when the host runtime and model weights are
   available; otherwise record the precise block.
5. Commit, push, open a stacked PR against PR #312's branch, and report CI status
   and any remaining runtime gaps.
