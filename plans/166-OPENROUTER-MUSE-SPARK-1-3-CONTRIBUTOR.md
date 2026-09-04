# OpenRouter Muse Spark 1.3 Contributor

## Goal

Add OpenRouter's stable `meta/muse-spark-1.3-contributor` route as an explicit
hosted Geist model without changing the existing online-agent architecture.

## Evidence and gate

- OpenRouter model API and endpoint API, checked 2026-09-04:
  - https://openrouter.ai/api/v1/models
  - https://openrouter.ai/api/v1/models/meta/muse-spark-1.3-contributor/endpoints
- OpenRouter model page, checked 2026-09-04:
  https://openrouter.ai/meta/muse-spark-1.3-contributor
- Meta release and evaluation methodology, published 2026-09-02:
  - https://research.meta.ai/blog/introducing-muse-spark-1-3
  - https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology
- Independent evaluation, published 2026-09-02:
  https://artificialanalysis.ai/articles/muse-spark-1-3

Gate score: 22/25 (capability value 5, evidence quality 4, Geist fit 5,
operational safety 3, implementation confidence 5). The safety score reflects
the contributor tier's training permission, Meta's 30-day retention, and its
single upstream endpoint. It must not be used for confidential workloads.

## Verified metadata

- Stable model ID: `meta/muse-spark-1.3-contributor`
- Provider/backend: OpenRouter / OpenAI-compatible; hosted only
- Context: 1,048,576 tokens
- Maximum completion: 943,718 tokens
- Input/output: text, image, video, file, and audio to text; OpenRouter warns
  that audio understanding is not yet fully supported
- Capabilities: streaming, tools/tool choice, reasoning, response format, and
  structured outputs
- Reasoning: mandatory, default effort `medium`
- Unsupported Geist request fields: `n`, `frequency_penalty`,
  `presence_penalty`, and `stop`
- Price: $0.10/M input, $0.20/M output, $0.002/M cache read
- Parameters/model size: not disclosed; leave unset
- Privacy: OpenRouter content logging is opt-in, but contributor prompts and
  outputs may be used to improve Meta products; Meta retains prompts for 30
  days and this endpoint is not eligible for ZDR routing

## Implementation

1. Add the hosted `ModelSpec` beside the existing Muse contributor route.
2. Reuse the existing OpenRouter provider and `OPENROUTER_API_KEY` resolution.
3. Cover catalog metadata, local-load rejection, factory routing, native tool
   preservation, and model-specific request constraints.
4. Update model/provider documentation with limits, pricing, evidence, and
   privacy caveats.

## Validation

- Targeted Ruff formatting/checks and mypy for changed Python files.
- Focused catalog and online-agent pytest suites.
- Affected frontend model-selection tests.
- Native API catalog smoke without a paid inference call.
- Geist pre-push test loop in proportion to this catalog-only change; report
  Docker/browser/native checks that are skipped or blocked.
