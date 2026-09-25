# OpenRouter GPT-6 Luna integration

## Goal

Expose the stable `openai/gpt-6-luna` route as a hosted OpenRouter model without
allowing it to fall through to local loading. Preserve native tool calling in
Geist's Chat Completions integration by using the model's documented
non-reasoning mode.

## Evidence checked on 2026-09-25

- OpenRouter's live model API lists the stable ID, 1,050,000-token context,
  128,000-token output limit, text/image/file input, optional reasoning,
  structured output, tools, and $0.10/M input plus $0.50/M output pricing:
  https://openrouter.ai/api/v1/models
- OpenRouter exposes OpenAI, Azure, and Amazon Bedrock routes and lists three
  Azure routes in its ZDR inventory:
  https://openrouter.ai/api/v1/models/openai/gpt-6-luna-20260922/endpoints
  https://openrouter.ai/api/v1/endpoints/zdr
- OpenAI documents vision, streaming, structured output, function calling, the
  same token limits and pricing, and says Chat Completions function calling is
  supported only with `reasoning_effort=none`:
  https://developers.openai.com/api/docs/models/gpt-6-luna
- Artificial Analysis independently reports a 37 Intelligence Index and about
  145 output tokens/second at max reasoning. Its non-reasoning result scores 18
  at roughly 124 output tokens/second:
  https://artificialanalysis.ai/models/releases/gpt-6-luna
- DynoTable reports 93.02% across its 201-case production agent suite, three
  runs per case, and a measured $0.000246 per call. Electricity Bench is more
  cautious: it reports a D overall grade and tier 2/5 on its real-world issue
  ladder, supporting Luna's role as a scoped, high-volume model rather than a
  flagship replacement:
  https://dynotable.com/blog/gpt-5-6-luna-benchmark
  https://electricitybench.com/models/codex-cli-gpt-6-luna/
- OpenAI states API data is not used for training by default, Chat Completions
  abuse-monitoring content may be retained for up to 30 days, and approved ZDR
  configurations exclude customer content from those logs:
  https://developers.openai.com/api/docs/guides/your-data

## Gate

| Dimension | Score | Reason |
| --- | ---: | --- |
| Capability value | 4 | Strong scoped-agent value, long context, vision, and native tools, but mixed complex-coding evidence |
| Evidence quality | 5 | First-party model card plus multiple independent exact-model evaluations |
| Geist fit | 5 | Stable OpenRouter ID and existing OpenAI-compatible OnlineAgent path |
| Operational safety | 5 | Multiple healthy routes and explicit ZDR-eligible Azure endpoints |
| Implementation confidence | 4 | Catalog constraints suffice, but tool use requires forced non-reasoning mode |
| **Total** | **23/25** | Clears the 20/25 gate with no score below 3 |

## Implementation

1. Add one explicit hosted `ModelSpec` with provider `openrouter`, backend
   `openai_compatible`, verified limits and capabilities, and no undisclosed
   parameter-count or family attribution.
2. Force reasoning effort `none` so Geist's Chat Completions tool path remains
   supported, and filter generic parameters absent from OpenRouter's supported
   parameter list.
3. Cover exact catalog metadata, local-load rejection, provider routing,
   OpenRouter key resolution, request shaping, and API exposure with focused
   tests.
4. Document pricing, reasoning/tool behavior, privacy defaults, mixed
   evaluation evidence, and the need to enforce OpenRouter ZDR for confidential
   workloads.

## Validation

- Run focused Ruff/format and mypy checks for changed Python files.
- Run focused catalog, factory-routing, OnlineAgent, and model API tests.
- Run affected frontend model-catalog tests.
- Exercise the native model-discovery API and Docker/browser smoke paths when
  feasible, without making a paid inference call.
- Run the Geist pre-push test loop, review the diff, commit, push, and open one
  ready PR against `main`.
