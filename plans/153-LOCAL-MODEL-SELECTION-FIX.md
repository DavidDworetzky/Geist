# Local Model Selection Fix

## Objective

Make the Settings model selector reflect the stored local model exactly and ensure the selected model uses its compatible local runner.

## Implementation

1. Define canonical local model identifiers and legacy aliases in a lightweight shared Python module.
2. Return and persist canonical model identifiers through the user-settings service.
3. Use canonical identifiers in the offline model catalog and frontend fallback catalog.
4. Normalize legacy identifiers before the existing catalog-based runner inference runs.
5. Preserve the generic open-weight model support and explicit runner overrides already present on `main`.

## Tests

1. Add unit coverage for legacy model-ID normalization and canonical settings updates.
2. Update frontend tests to assert the stored default matches an actual selector option and that Qwen selection is saved.
3. Update agent configuration tests to assert automatic runner inference remains enabled.
4. Run focused Python and frontend tests, then smoke-test the native backend and Settings selector in the browser.

## Out of Scope

- Downloading or deleting model weights from the UI.
- Supporting model architectures that do not have a registered local runner.
- Downloading model weights during tests.
