# OpenRouter Pareto 26.9

## Goal

Replace the retired `stealth/union-alpha` OpenRouter preview with its stable
`unbiased/pareto` release so Geist users do not select a model ID that is no
longer present in OpenRouter's live catalog.

## Research gate

OpenRouter added `unbiased/pareto` at 2026-09-17T23:02:58Z, after the previous
daily scout. The stable release is the disclosed identity of Union Alpha, which
Geist added as a preview before that identity was public.

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Capability value | 4/5 | Multimodal agentic/coding model with native tools and long output; provider results are strong but not category-leading everywhere. |
| Evidence quality | 3/5 | OpenRouter metadata and a reproducible provider harness are available; independent launch-day evidence is limited to a small external evaluation. |
| Geist fit | 5/5 | Exact OpenRouter ID uses the existing OpenAI-compatible OnlineAgent path and restores the retired preview entry. |
| Operational safety | 3/5 | Provider says no training, but retains requests for 30 days and the sole endpoint is not ZDR-eligible. |
| Implementation confidence | 5/5 | The stable route keeps the preview's request contract and needs only catalog, tests, and documentation changes. |
| **Total** | **20/25** | Clears the gate with no dimension below 3. |

## Verified metadata

- Stable OpenRouter ID: `unbiased/pareto`
- Provider/backend: `openrouter` / `openai_compatible`
- Context/output: 262,144 / 131,072 tokens
- Inputs/output: text and image to text
- Supported controls: `max_tokens`, `temperature`, `top_p`, `tools`, and
  automatic `tool_choice`
- Pricing checked September 18, 2026: $2.50/M input, $7.50/M output, and
  $0.25/M cached input
- Route: one Unbiased endpoint; 30-day retention, no training, and absent from
  OpenRouter's ZDR inventory

## Implementation

1. Replace the retired Union Alpha preview ModelSpec with the stable Pareto
   ModelSpec, retaining verified tool calling, vision, streaming, limits, and
   unsupported-parameter filtering.
2. Update focused catalog, factory routing, OnlineAgent completion, and streamed
   native-tool tests to use the stable ID.
3. Replace preview documentation with the stable release, pricing, evidence
   boundaries, and a clear prohibition on confidential workloads.
4. Verify the model appears in API/UI catalog data, routes through OpenRouter,
   resolves `OPENROUTER_API_KEY`, and cannot fall through to local loading.

## Validation plan

- Ruff lint and format checks on changed Python files
- mypy on the catalog, OnlineAgent, and factory routing files
- focused catalog, OnlineAgent, native-tool, settings, and model API tests
- affected frontend model-selection tests
- native API discovery smoke without a paid inference request
- changed-file pre-commit checks, diff review, and Docker smoke where feasible

Native MLX inference is not applicable because this is a hosted-only model and
does not change local loading or model weights.

## Validation evidence

- Ruff lint/format and focused mypy: passed.
- Focused catalog, factory-routing, OnlineAgent, and native-tool suites:
  182 passed after initializing the isolated SQLite database.
- Frontend `AgentConfigSection` suite in Docker: 19 passed; existing jsdom
  network/open-handle warnings only.
- Native API smoke: `/health/ready` and `/api/v1/models/` returned 200 with the
  exact Pareto metadata.
- Isolated Docker smoke: backend healthy; frontend and proxied models API
  returned 200; startup logs had no application errors.
- Browser smoke: Pareto appeared under OpenRouter with the expected limits and
  capabilities; browser console had no errors.
- Changed-file pre-commit suite: passed, including Ruff, mypy, Bandit, and
  staged secret scanning.
- Reused Docker backend image did not include `pytest`; the required
  containerized test command was attempted but blocked before test collection.
- No paid inference request was made. Native MLX testing was not applicable.

## Sources

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/unbiased/pareto
- https://openrouter.ai/api/v1/models/unbiased/pareto-20260917/endpoints
- https://openrouter.ai/providers/
- https://openrouter.ai/api/v1/endpoints/zdr
- https://openrouter.ai/docs/guides/privacy/data-collection
- https://openrouter.ai/docs/guides/features/zdr
- https://unbiased.ai/model-card/
- https://unbiased.ai/how/
- https://unbiased.ai/terms/
- https://aibenchy.com/model/unbiased-pareto-none/
