# OpenRouter Claude Sonnet 5.5

## Goal

Add Anthropic Claude Sonnet 5.5 as an explicit hosted OpenRouter model without
changing Geist's provider architecture or allowing the model to enter a local
loading path.

## Verified contract (2026-09-29)

- Stable OpenRouter ID: `anthropic/claude-sonnet-5.5`
- Canonical route: `anthropic/claude-sonnet-5.5-20260928`
- Provider/backend/local: `openrouter` / `openai_compatible` / `false`
- Limits: 1,000,000-token context and 128,000-token maximum completion
- Modalities: text, image, and file input; text output
- Capabilities: streaming, mandatory reasoning, native tools and tool choice,
  and JSON-schema structured output
- OpenRouter pricing: $2/M input, $10/M output, $0.20/M cache reads, and
  $2.50/M five-minute cache writes
- Parameters Geist must omit: `n`, `temperature`, `top_p`,
  `frequency_penalty`, and `presence_penalty`; the aggregate route advertises
  temperature only through its Azure upstreams, while stop remains portable
- Parameter count and architecture size are undisclosed and must remain unset

Sources:

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/anthropic/claude-sonnet-5.5
- https://openrouter.ai/api/v1/models/anthropic/claude-sonnet-5.5-20260928/endpoints
- https://openrouter.ai/api/v1/endpoints/zdr
- https://openrouter.ai/providers/
- https://www.anthropic.com/claude-sonnet-5-5
- https://www.vals.ai/models/anthropic_claude-sonnet-5-5
- https://electricitybench.com/models/claude-code-claude-sonnet-5-5/

## Scout gate

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Capability value | 5/5 | Strong current coding and knowledge-work results with a materially faster, lower-cost routine-work profile than Opus 5.5. |
| Evidence quality | 5/5 | Exact OpenRouter/Anthropic metadata plus current Vals AI and hands-on coding-agent measurements. |
| Geist fit | 5/5 | Existing OpenRouter OpenAI-compatible path supports the exact route, native tools, streaming, and structured output. |
| Operational safety | 4/5 | Eight healthy routes and no-training providers, including zero-retention upstreams, but the exact route is not yet present in OpenRouter's ZDR inventory and default routing may retain data for 30 days. |
| Implementation confidence | 5/5 | The existing Opus 5.5 catalog and request-filtering path covers the same integration contract. |
| **Total** | **24/25** | Clears the 20/25 gate with no score below 3. |

## Implementation

1. Add one explicit `ModelSpec` with verified limits, capabilities, mandatory
   high reasoning, unsupported request fields, and no parameter-count guess.
2. Extend catalog, local-load rejection, factory routing, API exposure, API-key
   resolution, native-tool, and request-payload tests.
3. Add a focused frontend selector test proving the API-delivered model appears
   under OpenRouter.
4. Document pricing, supported behavior, benchmark evidence, and privacy caveats.

## Privacy and reliability

OpenRouter lists Anthropic and Claude Platform on AWS as no-training with
30-day retention, while Google Vertex, Amazon Bedrock, and Azure are listed as
no-training and zero retention. The exact Sonnet 5.5 route was absent from the
ZDR endpoint API on 2026-09-29, so ordinary or `zdr: true` routing must not be
represented as ready for confidential workloads. Re-check the live ZDR
inventory or explicitly pin an approved zero-retention provider before such
use. OpenRouter endpoint observations are not availability guarantees.

## Validation

- Ruff lint and format check on changed Python
- mypy on catalog, OnlineAgent, and factory
- focused catalog/factory/API/OnlineAgent pytest selections
- affected frontend selector test
- native model-discovery API smoke
- change-scoped pre-commit checks and `git diff --check`
- Docker startup/log/curl/browser loop when the local daemon is available

No paid inference call is required, and no credential file or credential value
may be read or printed.
