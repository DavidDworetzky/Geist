# Local and Online Model Taxonomy

## Goal

Expose only two model execution choices: Local models managed and run by Geist, and Online models reached through supported vendor APIs. Protocol details such as loopback HTTP and OpenAI-compatible payloads must remain internal adapter concerns.

## Implementation

1. Present the persisted `local` and `online` agent values directly as the two inference modes.
2. Limit user-selectable provider discovery to `offline` local models and providers with supported online API endpoints.
3. Remove legacy Hugging Face and self-hosted groups from Models and Settings selection without removing internal model-family safety metadata.
4. Keep Hugging Face repository IDs exclusively as source metadata for managed local artifacts.
5. Hide architecture-incompatible local artifacts and normalize presentation to four user-facing phases: Not installed, Installing with progress, Installed, or Install failed.
6. Let the global selector show every compatible local artifact. A deliberate selection persists immediately and starts installation when needed; merely loading a configured default never starts a download.
7. Reconcile persisted installation state with disk, converging missing or invalid managed artifacts idempotently to `not_installed`. A new install clears invalid managed files before the capacity check, interrupted installs become retryable failures, and interrupted cancellations finish as not installed. Preflight free space before queueing a known-size artifact, and preserve the backend's specific storage error in the UI.
8. Start runtime loading only after installation succeeds. Chat stays blocked during installation/loading, uses compact state language, and reserves detailed copy for actionable failures.

## Validation

- Verify the models API exposes local plus supported online providers, excluding Hugging Face and self-hosted groups.
- Verify Models and Settings use Local/Online terminology and preserve selection behavior.
- Verify selector-driven installs, progress presentation, insufficient-storage preflight, cancellation, install failure, installed selection, runtime loading, and runtime failure.
- Run focused backend catalog tests, frontend model/settings tests, production build, and browser smoke checks.
