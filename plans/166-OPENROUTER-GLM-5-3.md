# OpenRouter GLM 5.3 integration plan

## Goal

Expose the stable OpenRouter model ID `z-ai/glm-5.3` as a hosted Geist model
without adding a dependency, a model-specific runner, or a local-loading path.

## Verified metadata (2026-09-08)

- Provider/backend: `openrouter` / `openai_compatible`
- Modalities: text input and text output
- Model context: 1,310,720 tokens in OpenRouter's model API
- Conservative output limit: 131,072 tokens, matching Z.AI's documented 128K
  maximum and OpenRouter's model-page FAQ
- Capabilities: mandatory reasoning, streaming, native tools, and structured output
- Reasoning contract: `low`, `high`, and `max`, with `max` as the default
- Unsupported Geist default: `n`
- Pricing: from $1.113/M input and $3.498/M output on the live OpenRouter page
- Operations: 28 OpenRouter endpoints; 23 endpoint records are currently ZDR
  eligible, including 20 active routes
- Privacy: OpenRouter prompt/response retention is opt-in, and ZDR routing can
  restrict downstream processing to endpoints that retain no request content

OpenRouter's aggregate API currently reports a 943,718-token top-provider
completion ceiling, but Z.AI's documentation and OpenRouter's model-page FAQ
both state 128K. The catalog uses the conservative, independently corroborated
131,072-token value instead of the larger route-specific ceiling.

## Implementation

1. Add one explicit hosted `ModelSpec` using the existing OpenRouter provider.
2. Apply mandatory `max` reasoning and omit the unsupported `n` parameter while
   retaining native tool payloads.
3. Cover exact metadata, API/UI discovery, factory routing, local-load rejection,
   OpenRouter key resolution, and request constraints with focused tests.
4. Document selection, pricing, provider variation, and ZDR guidance.

## Validation

1. Run Ruff lint/format checks and mypy for changed Python files.
2. Run focused model-catalog, OnlineAgent, route, and frontend tests.
3. Run a native model API smoke without making a paid inference call.
4. Follow the Geist pre-push test loop in proportion to this catalog-only change,
   reporting Docker, browser, or native checks that are blocked or not applicable.
5. Review the staged diff for correctness, security, and performance before push.

## Sources

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/z-ai/glm-5.3
- https://openrouter.ai/api/v1/models/z-ai/glm-5.3-20260816/endpoints
- https://openrouter.ai/api/v1/endpoints/zdr
- https://openrouter.ai/providers
- https://openrouter.ai/docs/guides/privacy/data-collection
- https://openrouter.ai/docs/guides/features/zdr
- https://docs.z.ai/guides/llm/glm-5.3
- https://github.com/zai-org/GLM-5
- https://artificialanalysis.ai/models/glm-5-3/
