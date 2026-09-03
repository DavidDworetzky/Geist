# Local and Online Model Taxonomy

## Goal

Expose only two model execution choices: Local models managed and run by Geist, and Online models reached through supported vendor APIs. Protocol details such as loopback HTTP and OpenAI-compatible payloads must remain internal adapter concerns.

## Implementation

1. Present the persisted `local` and `online` agent values directly as the two inference modes.
2. Limit user-selectable provider discovery to `offline` local models and providers with supported online API endpoints.
3. Remove legacy Hugging Face and self-hosted groups from Models and Settings selection without removing internal model-family safety metadata.
4. Keep Hugging Face repository IDs exclusively as source metadata for managed local artifacts.
5. Hide architecture-incompatible local artifacts, use explicit download-state labels, and never start a model download just because it is configured as the default.
6. Make the compatible local artifact inventory the single local-model selection path; Settings reports the configured model and links back to that inventory.

## Validation

- Verify the models API exposes local plus supported online providers, excluding Hugging Face and self-hosted groups.
- Verify Models and Settings use Local/Online terminology and preserve selection behavior.
- Run focused backend catalog tests, frontend model/settings tests, production build, and browser smoke checks.
