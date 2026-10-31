# OpenRouter Grok 4.6 integration plan

## Goal

Expose the stable OpenRouter model ID `x-ai/grok-4.6` as a hosted Geist model
without adding dependencies or allowing local-loading fallback.

## Evidence and gate

Evidence checked on 2026-08-21:

- OpenRouter model API and page: https://openrouter.ai/x-ai/grok-4.6
- OpenRouter endpoint API: https://openrouter.ai/api/v1/models/x-ai/grok-4.6-20260810/endpoints
- xAI launch and benchmark report (2026-08-12): https://x.ai/news/grok-4-6
- xAI API documentation: https://docs.x.ai/developers/grok-4-6
- Artificial Analysis independent evaluation: https://artificialanalysis.ai/models/grok-4-6
- OpenRouter ZDR policy: https://openrouter.ai/docs/guides/features/zdr

Gate score: capability value 5, evidence quality 5, Geist fit 5,
operational safety 4, implementation confidence 5; total 24/25.

## Implementation

1. Add OpenRouter and Grok 4.6 catalog metadata with the verified model ID,
   500K context, capabilities, mandatory reasoning, unsupported request
   parameters, and no guessed output limit or parameter count.
2. Route the model through `OnlineAgent`, resolve `OPENROUTER_API_KEY`, preserve
   native tool calling, and apply cataloged request constraints.
3. Expose OpenRouter through the existing model API/UI provider path and add
   focused catalog, factory, request-payload, and frontend tests.
4. Document setup and the requirement to enable ZDR routing for confidential
   workloads.

## Validation

- Focused Ruff format/lint and mypy for changed Python files.
- Focused pytest suites for model catalog, factory routing, and OnlineAgent.
- Focused React test for provider/model visibility.
- Native model API smoke without a paid inference request.
- Geist pre-push test loop in proportion to the hosted catalog change.
