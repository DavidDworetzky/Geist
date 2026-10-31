# OpenRouter Qwen3.8 Flash integration plan

## Goal

Expose the stable OpenRouter model ID `qwen/qwen3.8-flash` as a hosted Geist
model without adding a dependency or a model-specific runner.

## Verified sources (2026-08-27)

- OpenRouter model API: https://openrouter.ai/api/v1/model/qwen/qwen3.8-flash
- OpenRouter endpoint API: https://openrouter.ai/api/v1/models/qwen/qwen3.8-flash/endpoints
- OpenRouter model page: https://openrouter.ai/qwen/qwen3.8-flash
- OpenRouter ZDR endpoint API: https://openrouter.ai/api/v1/endpoints/zdr
- Qwen release post: https://qwen.ai/blog?id=qwen3.8-flash-next
- Artificial Analysis evaluation: https://artificialanalysis.ai/models/qwen3-8-flash-next
- Alibaba Model Studio privacy FAQ: https://www.alibabacloud.com/help/en/model-studio/faq-about-alibaba-cloud-model-studio

## Scope

1. Add an explicit OpenRouter-backed `ModelSpec` with the live 1,000,000-token
   context, 131,072-token output limit, multimodal input, reasoning, streaming,
   native tools, and hosted-only routing metadata.
2. Omit the unsupported `n` request parameter while retaining native tool
   payloads and optional reasoning behavior.
3. Cover catalog/API visibility, factory routing, local-load prevention,
   OpenRouter key resolution, and request constraints with focused tests.
4. Document the route, price, single-provider reliability caveat, and the fact
   that the current Alibaba endpoint is absent from OpenRouter's ZDR list.

## Validation

- Ruff lint/format and mypy for changed Python files.
- Focused model catalog, factory, API, and OnlineAgent tests.
- Affected AgentConfigSection frontend tests.
- Native API model-list smoke without making a paid inference call.
- Geist pre-push test loop checks in proportion to this catalog-only change.
