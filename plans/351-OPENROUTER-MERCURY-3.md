# OpenRouter Mercury 3

## Goal

Add the stable `inception/mercury-3` route as a hosted OpenRouter model using
Geist's existing OpenAI-compatible `OnlineAgent`. Keep it out of every local
loading path, preserve native tools and optional reasoning, and add no provider
plumbing or third-party dependency.

## Scout results reviewed on 2026-10-20

OpenRouter's live standard model API listed 468 IDs. No model had an API
creation timestamp after the prior-run cutoff (`2026-10-19T15:02:19.748Z`).
Mercury 3 nevertheless received a full review because its canonical slug and
provider launch are dated October 19, and its route, documentation, and
independent hands-on evaluations became available after the cutoff. MiniMax
M3.1 Flash was rechecked because the saved prior-run memory is missing.

Neither exact ID appears on `origin/main` or in the title, branch, body, or diff
of any of the 26 open pull requests.

| Candidate | Capability | Evidence | Geist fit | Safety | Confidence | Total | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mercury 3 | 4 | 4 | 5 | 4 | 5 | 22 | Selected |
| MiniMax M3.1 Flash Free | 4 | 5 | 5 | 2 | 4 | 20 | Rejected: no callable endpoint in the live endpoint API; retaining, non-ZDR provider policy |

## Verified facts

- OpenRouter exposes the stable `inception/mercury-3` ID with a 1,000,000-token
  context window, 131,072-token maximum output, text input/output, optional
  reasoning enabled by default at medium effort, native tools/tool choice,
  response-format and JSON-schema structured output, and streaming-compatible
  chat parameters. Current launch pricing is $0.10/M input, $0.35/M output, and
  $0.01/M cache reads:
  https://openrouter.ai/api/v1/models
  https://openrouter.ai/inception/mercury-3
- The exact endpoint API listed one active Inception route with 100% 30-minute
  uptime and approximately 99.92% one-day uptime. The route appeared in the
  official ZDR endpoint inventory. OpenRouter's provider directory identifies
  Inception as no-training and zero-retention:
  https://openrouter.ai/api/v1/models/inception/mercury-3-20261019/endpoints
  https://openrouter.ai/api/v1/endpoints/zdr
  https://openrouter.ai/providers
- Inception's October 19 launch documentation describes Mercury 3 as a closed,
  diffusion-based model with an OpenAI-compatible API and publishes a list price
  of $0.20/M input and $0.70/M output, currently discounted by 50%. It discloses
  neither total nor activated parameter count:
  https://www.inceptionlabs.ai/blog/introducing-mercury-3
- Inception's general direct-service terms permit training unless the account
  opts out, while its enterprise offering advertises configurable retention.
  This integration relies on OpenRouter's endpoint-specific ZDR policy; users
  should still enforce ZDR routing and re-check policy before confidential use:
  https://www.inceptionlabs.ai/docs/terms-of-use
  https://www.inceptionlabs.ai/enterprise
  https://openrouter.ai/docs/guides/features/zdr

## Provider claims

Inception reports 1,068 output tokens per second on NVIDIA GPUs and says Mercury
3 leads GPT-6 Luna and Claude Haiku 5.5 on 9 of 12 launch-table evaluations.
Published examples include IFBench 76.8, SWE-Bench Atlas 51.1, and
tau3-bench Banking 30.2. It also reports 50% completion on 50 live Mind2Web
navigation tasks in a production partner harness. These are provider-run or
provider-published partner results, not independent reproductions.

## Independent evidence and community reports

- Bito's October 20 coding-agent evaluation scored Mercury 3 at 76% on short
  sessions and 65% on long multi-turn sessions. A long session completed in
  5.8 minutes, versus 17.2 for Claude Opus 5.5, but quality trailed the tested
  Claude models on long work:
  https://bito.ai/blog/inception-mercury-3-diffusion-model-coding-benchmark/
- Spring Prompt tested the exact OpenRouter ID on October 19-20. Reasoning-off
  responses were exceptionally fast, while default reasoning sometimes paused
  for more than a minute; business-analysis scores ranked near the bottom of
  its current comparison set:
  https://springprompt.com/blog/mercury-3-review
- The Neuron ran eight OpenRouter API spot checks across writing, arithmetic,
  explanation, coding, hallucination resistance, and logic. The reviewer found
  the API fast and useful but explicitly did not reproduce the provider's peak
  throughput claim. This is supporting hands-on evidence, not a controlled
  benchmark:
  https://www.theneuron.ai/news/mercury-3-review-diffusion-language-models/

## Inference

Mercury 3 materially improves Geist/Pitchblend for latency-sensitive,
tool-using subagents and short coding or search steps: it combines a million
tokens of context, native tools, schema output, very low launch pricing, and a
ZDR route. Independent evidence does not support making it the default for
long-horizon planning. The single provider route also limits redundancy, so
ordinary retry/fallback behavior remains important.

MiniMax M3.1 Flash has strong exact-model evidence, but its official endpoint
API returned an empty endpoint list at review time. Its sole MiniMax policy is
prompt-retaining and the exact ID was absent from the ZDR inventory. It is not
currently a stable, confidential-workload-safe addition even though the model
catalog entry and public page remain visible.

## Implementation

1. Add one explicit hosted `ModelSpec` with verified limits and capabilities,
   no parameter-count fields, `provider="openrouter"`,
   `backend="openai_compatible"`, and `local=False`.
2. Filter only generic request parameters absent from the exact OpenRouter
   contract: `n`, `top_p`, `frequency_penalty`, and `presence_penalty`. Preserve
   native tools, tool choice, stop, temperature, structured output, and optional
   reasoning.
3. Add catalog, local-fallback, factory-routing, API/UI visibility, key
   resolution, request-shaping, documentation, and frontend-selection coverage.

## Validation

Run targeted Ruff format/lint, mypy, focused catalog/factory/OnlineAgent/API
pytest, the affected frontend test, native FastAPI discovery, applicable
pre-push hooks, and Docker/log/curl/browser checks when available. Native MLX is
not applicable to this hosted-only model. Do not make a paid inference call or
read credential files.
