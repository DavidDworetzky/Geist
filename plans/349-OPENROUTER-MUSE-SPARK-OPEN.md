# OpenRouter Muse Spark Open

## Goal

Add the stable `meta/muse-spark-open` OpenRouter model to Geist as a hosted-only
OpenAI-compatible option without changing the agent architecture or adding a
dependency.

## Evidence and gate

Verified October 15, 2026:

- OpenRouter exposes the stable ID with 1,048,576 context tokens, a 943,718-token
  output ceiling, multimodal input, mandatory reasoning, native tools, structured
  output, and four routed providers.
- The model is in OpenRouter's ZDR-filtered catalog. Together, Nebius, Fireworks,
  and Baseten are listed by OpenRouter as no-training and zero-retention.
- Meta discloses 816B total and 42B active parameters and publishes exact agent,
  coding, multimodal, and safety evaluation results in the model card.
- Artificial Analysis reports an Intelligence Index of 39 at xhigh effort.
  Plotly's independent 43-task analytics/tool-use run passed 38 tasks (88%).

Gate: capability 4, evidence quality 5, Geist fit 5, operational safety 5,
implementation confidence 5 = 24/25. Every dimension is at least 3.

Sources:

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/meta/muse-spark-open
- https://openrouter.ai/api/v1/models/meta/muse-spark-open-20261015/endpoints
- https://openrouter.ai/api/v1/models?zdr=true
- https://openrouter.ai/providers
- https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B
- https://research.meta.ai/blog/muse-spark-open
- https://artificialanalysis.ai/changelog
- https://plotly.com/blog/meta-muse-spark-open-first-look/

## Implementation

1. Add one explicit hosted `ModelSpec` using the existing OpenRouter provider.
2. Preserve the model's native tools and structured-output fields, remove only
   unsupported `n`, and apply OpenRouter's mandatory default `medium` reasoning.
3. Cover catalog metadata, hosted factory routing, API/UI discovery, key
   resolution, and request shaping.
4. Document exact metadata, pricing, evidence, and ZDR requirements.

## Validation

Run targeted Ruff format/lint and mypy, focused catalog/factory/OnlineAgent/API
pytest, the affected frontend test, a native API smoke, and the applicable
Geist pre-push loop. Do not make a paid inference call.
