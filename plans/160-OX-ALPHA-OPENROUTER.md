# Ox Alpha OpenRouter Integration Plan

## Goal

Expose OpenRouter's hosted Ox Alpha preview as an online Geist model so
Pitchblend can select it without attempting a local model load.

## Implementation

1. Add `stealth/ox-alpha` to the model catalog as an OpenRouter-backed,
   OpenAI-compatible model with the context, output, multimodal, tool-calling,
   reasoning, and streaming capabilities published by OpenRouter.
2. Keep the model explicitly hosted and avoid attributing an undisclosed model
   family or parameter count.
3. Document the model alongside the existing OpenRouter API-key configuration.
4. Extend catalog, factory-routing, API-key, and native-tool capability tests.

## Validation

- Run focused model-catalog and online-agent tests.
- Run formatting and lint checks for the changed Python files.
- Verify the frontend provider/model catalog tests still pass.
- Exercise the external inference boundary only when an OpenRouter key is
  intentionally supplied; never read or print local credentials.
