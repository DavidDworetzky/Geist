# OpenRouter MiniMax M3.1 Flash

## Goal

Add the stable `minimax/minimax-m3.1-flash` route to Geist as an explicit
hosted-only model after OpenRouter added multiple providers and ZDR-eligible
endpoints on October 23, 2026.

## Selection gate

| Dimension | Score | Basis |
| --- | ---: | --- |
| Capability value | 4 | Strong multimodal agent, coding, analytics, and long-context results, with weaker long-horizon terminal evidence. |
| Evidence quality | 5 | Provider benchmarks plus current Vals, Plotly, and hands-on exact-model coding evaluations. |
| Geist fit | 5 | OpenAI-compatible chat, native tools, structured outputs, streaming, and no new dependency. |
| Operational safety | 5 | Stable ID, eight providers, six ZDR routes, and OpenRouter's available per-request ZDR enforcement. |
| Implementation confidence | 5 | Existing catalog, factory, API/UI, key resolution, and request-filtering paths cover the model. |
| **Total** | **24/25** | Clears the 20/25 gate with no score below 3. |

The model was absent from `origin/main` and all 26 open pull-request heads.
PR #397 mentioned it only as a previously rejected candidate, before the new
provider and ZDR routes appeared.

## Verified contract

- Provider/backend/local: `openrouter` / `openai_compatible` / `false`.
- Context/output: 1,048,576 / 524,288 tokens from OpenRouter's cataloged model contract.
- Input/output: text, image, and video input; text output.
- Capabilities: streaming, native tools and tool choice, JSON-schema structured output, and mandatory reasoning.
- Reasoning: `max` is the default; OpenRouter also advertises explicit `low`, `medium`, `high`, and `xhigh` efforts.
- Parameters: 428B total / 23B active, disclosed by MiniMax and OpenRouter.
- Routing: eight providers observed; six endpoints were in OpenRouter's ZDR inventory.
- Pricing: routed prices from $0.10/M input, $0.55/M output, and $0.01/M cache reads; MiniMax list price $0.15/$0.60/$0.015.

## Implementation

1. Add a hosted `ModelSpec` using the existing OpenRouter `ProviderSpec` and exact stable model ID.
2. Mark the route server-backed so factory inference cannot fall through to local loading.
3. Filter unsupported `n`, apply mandatory default max reasoning, and preserve native tools and structured outputs.
4. Extend catalog, factory-routing, API/UI, key-resolution, request-shaping, frontend, and documentation coverage.

## Validation

Run focused Ruff/format, mypy, catalog/factory/OnlineAgent pytest, the affected
frontend suite, native model API smoke, and the applicable Geist pre-push loop.
Use Docker/log/curl/browser smoke when the daemon is available. Do not make a
paid inference call or inspect credential files.
