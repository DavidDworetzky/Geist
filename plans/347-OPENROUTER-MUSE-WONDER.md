# OpenRouter Muse Wonder

## Goal

Add OpenRouter's stable `meta/muse-wonder` route as an explicit hosted Geist
model without changing the existing `OnlineAgent` architecture or adding a
dependency.

## Evidence and gate

Sources checked October 13, 2026:

- OpenRouter live catalog, model page, and exact endpoint metadata:
  - https://openrouter.ai/api/v1/models
  - https://openrouter.ai/meta/muse-wonder
  - https://openrouter.ai/api/v1/models/meta/muse-wonder-20261012/endpoints
- OpenRouter provider policy and ZDR inventory:
  - https://openrouter.ai/providers
  - https://openrouter.ai/api/v1/models?zdr=true
- Meta's October 12 release and research report:
  - https://about.fb.com/news/2026/10/introducing-muse-wonder-models/
  - https://research.meta.ai/blog/muse-wonder
- Independent Artificial Analysis evaluation, published October 12:
  https://artificialanalysis.ai/articles/muse-wonder
- Supporting launch-day hands-on test, published October 12:
  https://quantslant.com/muse-wonder/

Gate score: 23/25 (capability value 5, evidence quality 5, Geist fit 5,
operational safety 3, implementation confidence 5). Muse Wonder materially
improves the catalog's long-horizon agent and coding capability, and the exact
model has independent evaluation evidence. The operational score reflects its
single Meta endpoint, 30-day provider retention, and absence from OpenRouter's
ZDR inventory. It must not be used for confidential or regulated workloads.

## Verified metadata

- Stable model ID: `meta/muse-wonder`
- Canonical slug: `meta/muse-wonder-20261012`
- Release: October 12, 2026; not labeled preview
- Provider/backend: OpenRouter / OpenAI-compatible; hosted only
- Context: 1,048,576 tokens
- Maximum completion: 943,718 tokens
- Input/output: text, image, file/PDF, audio, and video to text
- Capabilities: streaming transport, native tools and tool choice, JSON-schema
  structured output, and mandatory configurable reasoning
- Reasoning: mandatory; OpenRouter default effort `medium`
- Unsupported Geist request fields: `n`, `frequency_penalty`,
  `presence_penalty`, and `stop`
- Price: $4/M input, $20/M output, $0.20/M cached input, and $2.50/1K web
  searches
- Parameters/model size: not disclosed; leave unset
- Privacy: OpenRouter lists Meta as no-training with 30-day retention; the
  exact model is absent from the ZDR-filtered catalog
- Reliability when checked: one active Meta endpoint, 99.93% one-day endpoint
  uptime, 99.33% routed availability over three days, 87 tokens/s P50, and
  2.54-second P50 latency

## Implementation

1. Add the hosted `ModelSpec` beside the existing OpenRouter Muse entry.
2. Reuse the existing OpenRouter provider, `OPENROUTER_API_KEY` resolution,
   native tool path, and request-constraint machinery.
3. Cover catalog metadata, local-load rejection, factory routing, API/UI
   exposure, key resolution, mandatory reasoning, and parameter filtering.
4. Update model/provider documentation with exact limits, evidence, pricing,
   and the 30-day-retention warning.

## Validation

- Targeted Ruff format/check and mypy for changed Python files.
- Focused catalog, factory-routing, API, and online-agent pytest suites.
- Affected frontend model-selection tests.
- Native FastAPI catalog smoke without a paid inference call.
- Applicable Geist pre-push checks plus Docker/log/curl/browser smoke when the
  local environment permits them; report blocked or skipped checks explicitly.
