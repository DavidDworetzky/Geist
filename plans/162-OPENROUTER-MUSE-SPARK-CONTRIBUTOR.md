# OpenRouter Muse Spark 1.2 Contributor integration plan

## Goal

Expose the stable OpenRouter model ID `meta/muse-spark-1.2-contributor` as a
hosted Geist model without adding dependencies or allowing local-loading
fallback.

## Evidence and gate

Evidence checked on 2026-08-22:

- OpenRouter live model API: https://openrouter.ai/api/v1/models?sort=newest
- OpenRouter endpoint API: https://openrouter.ai/api/v1/models/meta/muse-spark-1.2-contributor-20260805/endpoints
- OpenRouter Muse Spark 1.2 page: https://openrouter.ai/meta/muse-spark-1.2
- Meta launch report (2026-08-05): https://research.meta.ai/blog/introducing-muse-code-and-muse-spark-1-2
- Meta multimodal report (2026-08-20): https://research.meta.ai/blog/multimodal-intelligence-of-muse-spark-1-2
- Artificial Analysis evaluation: https://artificialanalysis.ai/models/muse-spark-1-2
- OpenRouter provider policies: https://openrouter.ai/providers
- OpenRouter ZDR endpoint API: https://openrouter.ai/api/v1/endpoints/zdr

Gate score: capability value 5, evidence quality 4, Geist fit 4,
operational safety 3, implementation confidence 4; total 20/25.

## Implementation

1. Add exact hosted catalog metadata without guessing a fixed output limit or
   parameter count.
2. Preserve OpenRouter native tool calling and apply the model's mandatory
   medium reasoning effort while removing unsupported request parameters.
3. Verify API/UI visibility, provider-key resolution, factory routing, and
   local-loading prevention with focused tests.
4. Document the upstream 30-day retention policy and prohibit confidential
   workloads because the endpoint does not support ZDR.

## Validation

- Focused Ruff format/lint and mypy for changed Python files.
- Focused pytest for catalog, factory routing, and OnlineAgent payloads.
- Focused React coverage for provider/model visibility.
- Native model API smoke without a paid inference request.
- Proportional Geist pre-push test loop checks.
