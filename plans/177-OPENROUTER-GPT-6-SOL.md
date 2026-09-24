# OpenRouter GPT-6 Sol integration

## Goal

Expose the stable `openai/gpt-6-sol` route as a hosted OpenRouter model without
allowing it to fall through to local loading. Preserve native tool calling in
Geist's Chat Completions integration by using the model's documented
non-reasoning mode.

## Evidence checked on 2026-09-23

- OpenRouter's live model API lists the stable ID, 1,050,000-token context,
  128,000-token output limit, text/image/file input, optional reasoning,
  structured output, tools, and $2/M input plus $10/M output pricing:
  https://openrouter.ai/api/v1/models
- OpenRouter exposes OpenAI, Azure, and Bedrock routes and lists Azure routes in
  its ZDR inventory:
  https://openrouter.ai/api/v1/models/openai/gpt-6-sol-20260922/endpoints
  https://openrouter.ai/api/v1/endpoints/zdr
- OpenAI documents vision, streaming, structured output, function calling, the
  same token limits and pricing, and says Chat Completions function calling is
  supported only with `reasoning_effort=none`:
  https://developers.openai.com/api/docs/models/gpt-6-sol
- Artificial Analysis independently reports a 28.09 Intelligence Index,
  13.13% Terminal-Bench 4.0, and about 104 output tokens/second in
  non-reasoning mode. Its max-reasoning evaluation reaches 47.53 and 43.94%:
  https://artificialanalysis.ai/models/gpt-6-sol-non-reasoning
  https://artificialanalysis.ai/models/gpt-6-sol
- OpenAI states API data is not used for training by default, Chat Completions
  abuse-monitoring content may be retained for up to 30 days, and approved ZDR
  configurations exclude customer content from those logs:
  https://developers.openai.com/api/docs/guides/your-data

## Gate

| Dimension | Score | Reason |
| --- | ---: | --- |
| Capability value | 5 | Strong independent coding/agent evaluation, long context, vision, and native tools |
| Evidence quality | 5 | First-party model card plus independent exact-mode evaluation |
| Geist fit | 5 | Stable OpenRouter ID and existing OpenAI-compatible OnlineAgent path |
| Operational safety | 5 | Multiple healthy routes and explicit ZDR-eligible Azure endpoints |
| Implementation confidence | 4 | Catalog constraints suffice, but tool use requires forced non-reasoning mode |
| **Total** | **24/25** | Clears the 20/25 gate with no score below 3 |

## Implementation

1. Add one explicit hosted `ModelSpec` with provider `openrouter`, backend
   `openai_compatible`, verified limits and capabilities, and no undisclosed
   parameter-count or family claims.
2. Force reasoning effort `none` so Geist's Chat Completions tool path remains
   supported, and filter generic parameters absent from OpenRouter's supported
   parameter list.
3. Cover exact catalog metadata, local-load rejection, provider routing,
   OpenRouter key resolution, request shaping, and API exposure with focused
   tests.
4. Document pricing, reasoning/tool behavior, privacy defaults, and the need to
   enforce OpenRouter ZDR for confidential workloads.

## Validation

- Run focused Ruff/format and mypy checks for changed Python files.
- Run focused catalog, factory-routing, OnlineAgent, and model API tests.
- Run affected frontend model-catalog tests.
- Exercise the native model-discovery API and Docker/browser smoke paths when
  feasible, without making a paid inference call.
- Run the Geist pre-push test loop, review the diff, commit, push, and open one
  ready PR against `main`.
