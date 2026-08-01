# Kimi K3 Online Inference

## Goal

Add Kimi K3 to Geist's online model catalog so it can be selected and routed
through Moonshot AI's OpenAI-compatible API.

## Scope

- Add the hosted `k3` model to the Moonshot provider catalog.
- Record the capabilities Geist exposes for K3: one-million-token context,
  vision, reasoning, function calling, and streaming.
- Keep the existing `MOONSHOT_API_KEY` and Moonshot endpoint configuration.
- Add focused catalog and agent-factory coverage for discovery and routing.

No database migration, new provider, frontend-specific model list, or local
runner implementation is required because the model API and UI are catalog
driven and K3 is server backed.

## Verification

- Run the focused model-catalog tests.
- Run Ruff on the changed Python files.
- Check the patch for whitespace errors.
- Exercise Docker startup and the model API when the local Docker environment
  has sufficient capacity; native MLX validation is not applicable to an
  online-only model.
