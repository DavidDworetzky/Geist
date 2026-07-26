# Native host platform-isolation follow-up

## Goal

Keep PR #296's native host, managed `llama-server` backend, MLX compatibility,
and PyInstaller packaging while making platform-specific process behavior
explicit, type-checkable, and testable on every supported host.

## Scope

1. Move Windows process creation and Job Object handling out of the common
   `llama-server` lifecycle module. Keep the common manager responsible for
   model-server state, health, cancellation, and backend fallback.
2. Keep POSIX process-group behavior behind the same small platform interface
   so the shared manager does not branch on Windows-only APIs.
3. Make managed-model unit tests deterministic instead of inheriting the host
   machine's GGUF/MLX support policy.
   Preserve `LOCAL_WEIGHTS_DIR` as a legacy MLX fallback, but let an explicit
   managed `artifact_id` take precedence.
4. Add lightweight Windows, macOS ARM64, and Linux CI checks for dependency
   resolution, type checking, native-host tests, and packaging input/argument
   validation. Full signed installers and real model inference remain outside
   this PR's CI scope.
5. Compare PR #295's provider credentials and model-download implementation
   with #296's artifact store. Do not merge #295 code until the overlapping
   contracts and UI behavior are reviewed with the user.

## Validation

- `uv run mypy .`
- Focused `pytest` for the process manager, local model manager, CLI, runtime
  configuration, loopback middleware, database upgrade, and sidecar builder.
- Existing frontend model tests when dependencies are already available.
- `git diff --check` and a final overlap audit against PRs #295, #297, and #298.

## Outcome

- Shared llama-server lifecycle code no longer imports Windows-only APIs.
- PyInstaller inputs include only the target platform's process module.
- Explicit managed MLX artifacts override the legacy weights-directory fallback.
- A three-platform native-host CI matrix covers type checking and focused tests.
- PR #295 remains a comparison source; its competing download service and
  provider credential persistence are not merged by default.

## Non-goals

- Replacing the out-of-process `llama-server` architecture.
- Removing or redesigning the MLX path.
- Building signed/notarized distributables in general PR CI.
- Combining #295's provider-key feature with the local-model decision before
  the feature comparison is agreed.
