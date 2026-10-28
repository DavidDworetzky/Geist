# OpenRouter GPT-6.1 Terra

## Goal

Add the stable `openai/gpt-6.1-terra` route to Geist's hosted model catalog
without changing the existing OpenAI-compatible architecture or adding a
dependency.

## Selection gate

Research refreshed on October 28, 2026 against OpenRouter's live catalog,
model pages, endpoint inventory, `origin/main`, and every open pull request.

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Capability value | 5 | 1.05M multimodal context, strong computer-use evidence, and competitive coding-agent results at half the token price of GPT-6.1 Sol |
| Evidence quality | 5 | OpenAI and OpenRouter contracts plus current Artificial Analysis and independent hands-on results |
| Geist fit | 5 | The stable OpenRouter ID exposes chat completions, streaming, tools, and structured output through the existing OnlineAgent |
| Operational safety | 3 | OpenAI reports no training, but the current routes retain prompts and are absent from OpenRouter's ZDR inventory |
| Implementation confidence | 5 | Exact request limits and unsupported defaults are published and match existing catalog-driven filtering |
| **Total** | **23/25** | Clears the required 20/25 gate with no dimension below 3 |

`openai/gpt-6.1-terra-pro` is deferred because it is the same underlying model
locked to expensive pro reasoning, OpenRouter recommends standard Terra for
everyday coding and agentic work, and no separate independent Pro evaluation is
available. The two `:batch` aliases are asynchronous and fail Geist's
interactive-use requirement.

## Verified contract

- Model ID: `openai/gpt-6.1-terra`
- Canonical slug: `openai/gpt-6.1-terra-20261026`
- Provider/backend/local: `openrouter` / `openai_compatible` / `false`
- Context/output: 1,050,000 / 128,000 tokens
- Input/output: text, image, and file input; text output
- Reasoning: optional; none through max, with medium enabled by default
- Capabilities: streaming, native tools and all tool-choice modes, and
  JSON-schema structured output
- Unsupported generic request fields: `n`, `temperature`, `top_p`,
  `frequency_penalty`, `presence_penalty`, and `stop`
- Standard pricing: $1/M input, $5/M output, $0.05/M cache read, and $1.25/M
  cache write; long prompts above 272,000 tokens use higher tiered rates
- Current routes: OpenAI Flex, standard, and Fast; all reported status `0` and
  100% one-day uptime when checked
- No parameter count or undisclosed architecture attribution

## Privacy and evidence

OpenRouter's provider directory lists OpenAI as no-training but
prompt-retaining. None of the exact Terra routes appeared in OpenRouter's ZDR
endpoint inventory on October 28, 2026. The model must not be recommended for
confidential or regulated workloads until an exact ZDR route appears.

Verified sources:

- OpenRouter catalog and exact endpoint API (checked 2026-10-28):
  https://openrouter.ai/api/v1/models and
  https://openrouter.ai/api/v1/models/openai/gpt-6.1-terra/endpoints
- OpenRouter model page (checked 2026-10-28):
  https://openrouter.ai/openai/gpt-6.1-terra
- OpenRouter provider and ZDR inventories (checked 2026-10-28):
  https://openrouter.ai/providers and
  https://openrouter.ai/api/v1/endpoints/zdr
- OpenAI launch post and model contract (published 2026-10-26):
  https://openai.com/index/introducing-gpt-6-1-terra/ and
  https://developers.openai.com/api/docs/models/gpt-6.1-terra
- Artificial Analysis release evaluation (published 2026-10-26):
  https://artificialanalysis.ai/models/releases/gpt-6-1-terra
- DualEntry hands-on accounting evaluation (published 2026-10-26):
  https://www.dualentry.com/blog/gpt-6-1-terra-accounting-benchmark

OpenAI's launch benchmark results are provider claims. Artificial Analysis and
DualEntry are independent evidence. Early community reports are supporting
evidence only and are not used to set the gate score.

## Implementation

1. Add one explicit hosted `ModelSpec` with exact limits and request filters.
2. Reuse the existing OpenRouter `ProviderSpec` and `OPENROUTER_API_KEY` path.
3. Cover catalog metadata, OpenRouter factory routing, local-fallback rejection,
   key resolution, request filtering, optional reasoning, native tools,
   structured output, and UI visibility.
4. Document capability, pricing, reliability, and the non-ZDR privacy caveat.

## Validation

1. Run focused Ruff lint/format and mypy on changed Python.
2. Run focused catalog, factory, OnlineAgent, route, and model API tests.
3. Run the affected frontend model-picker test.
4. Exercise native model discovery through the FastAPI route.
5. Run the Geist pre-push loop: applicable pre-commit hooks, Docker startup,
   backend/frontend logs, `localhost:3000` curl, and browser UI smoke when the
   local environment permits them.
6. Review the final diff for correctness, security, privacy language, and
   accidental secret exposure. Do not make a paid inference call.

Native `MLX_BACKEND=1` validation is not applicable because the model is
hosted-only and cannot enter a local runner.
