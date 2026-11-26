# OpenRouter Inkling 2 Mini

## Objective

Add the stable `thinkingmachines/inkling-2-mini` OpenRouter route as an
explicit hosted-only Geist model. Reuse the existing OpenAI-compatible
`OnlineAgent`, preserve native tool calling and caller-selected reasoning,
expose verified metadata to the API/UI, and prevent local-loader fallback.

Research was refreshed on November 21, 2026 against the prior scout cutoff of
November 20, 2026 at 16:09:39 UTC, `origin/main` at
`5f3e7c52a6e3686f3a699f9224dd2b2856ac78cb`, and all 39 open pull requests.
Neither Inkling 2 Mini nor DaisyRose 27B is present in the catalog or an open
PR. Existing model-related PRs were treated as in-flight work and not
duplicated.

## Gate

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Capability value | 4 | Independent coding and agent evaluations show Qwen3.8-27B-class results at much lower hosted cost, adding a useful efficient worker/verifier tier. |
| Evidence quality | 5 | Live OpenRouter contracts, the model card, Vals, and Artificial Analysis agree on the exact model's core capability profile. |
| Geist fit | 5 | Text/image input, optional reasoning, native tools, structured output, and streaming fit the existing `OnlineAgent`. |
| Operational safety | 5 | The stable identified model has two no-training, ZDR-eligible routes; route-dependent context limits remain documented. |
| Implementation confidence | 5 | The stable ID uses the existing OpenRouter provider/key path and only requires filtering unsupported `n`. |

**Total: 24/25; no dimension is below 3.**

`radixark/daisyrose-27b` scored 18/25: capability 3, evidence 3, Geist fit 4,
operational safety 4, and implementation confidence 4. It is deferred because
the largest agent gains remain provider-run, while an independent Plotly
hands-on evaluation scored it below its Qwen3.8-27B base on data analysis and
reported malformed tool calls. More established exact-model evaluation is
needed before catalog recommendation.

## Evidence

### Verified facts

- OpenRouter live model API, checked November 21, 2026: stable ID
  `thinkingmachines/inkling-2-mini`, canonical slug
  `thinkingmachines/inkling-2-mini-20261120`, creation time November 21, 2026
  at 02:19:45 UTC, text/image input, 1,048,576-token maximum context, 65,536
  maximum completion tokens, optional default-on reasoning, streaming Chat
  Completions, tools/tool choice, and structured output:
  https://openrouter.ai/api/v1/models
- Exact endpoint API, checked November 21, 2026: Prime Intellect FP4 at
  $0.075/M input, $0.30/M output, $0.015/M cache read, 1,048,576 context, and
  100% 30-minute uptime; io.net NVFP4 at $0.08/M input, $0.35/M output,
  $0.016/M cache read, 262,144 context, and 100% 30-minute uptime. Both support
  every generic Geist request parameter except `n`:
  https://openrouter.ai/api/v1/models/thinkingmachines/inkling-2-mini-20261120/endpoints
- OpenRouter's model page, checked November 21, 2026: 99.99% three-day
  reachability and 99.51% three-day availability across two routes:
  https://openrouter.ai/thinkingmachines/inkling-2-mini
- OpenRouter provider and ZDR inventories, checked November 21, 2026: Prime
  Intellect and io.net are no-training and zero-retention; both exact model
  endpoints are ZDR-eligible with `safety_retention: false`:
  https://openrouter.ai/providers and
  https://openrouter.ai/api/v1/endpoints/zdr
- Thinking Machines' November 20 model card: Apache-2.0, 26.4B total and 3.4B
  unique active parameters, 5.4B effective active parameters per token after
  loop passes, 1M context, text/image input, text output, and explicit safety
  and limitation disclosures:
  https://huggingface.co/thinkingmachines/Inkling-2-Mini
- Vals, published November 20, 2026: 46.46% overall, 77.92% Vibe Code Bench
  v1.1, 56.54% Finance Agent v2.1, 37.92% SWE-bench Multimodal, and 6.57%
  Terminal-Bench 4.0:
  https://www.vals.ai/models/thinkingmachines_inkling-2-mini
- Artificial Analysis, published November 20, 2026: Intelligence Index 20,
  $0.59 average index-task cost, nine minutes per task, and 161M output tokens
  across the index evaluation:
  https://artificialanalysis.ai/models/inkling-2-mini/

### Provider claims

- Thinking Machines reports 48.2% DeepSWE v1.1, 77.9% SWE-bench Pro v2 Hard,
  71.6% Terminal-Bench 2.1, 82.7% MCP Atlas, and 85.3% DeepSearchQA. Its model
  card also warns of hallucination, occasional instruction-following failures,
  and degraded long multi-turn performance:
  https://thinkingmachines.ai/news/inkling-2-mini/ and
  https://huggingface.co/thinkingmachines/Inkling-2-Mini

### Community reports

- Launch-day hands-on reports are mixed: several users report Qwen3.8-27B-like
  coding quality and strong code review, while others report heavy thinking,
  slow or immature local runtimes, and occasional agent-loop instability.
  These reports support qualification caution but do not set the gate:
  https://www.reddit.com/r/LocalLLM/comments/1y1ly94/opinions_on_inkling_2_mini/ and
  https://www.reddit.com/r/LocalLLaMA/comments/1y1yu0f/inkling_2_mini_seems_legit_as_reported_by_others/

### Inference

- Inkling 2 Mini materially improves Geist's low-cost hosted worker/verifier
  options because independent results approach the existing Qwen3.8-27B class
  while live token prices and effective compute are much lower.
- Preserve optional reasoning rather than forcing OpenRouter's `high` default;
  the live contract supports `none`, `minimal`, `low`, `medium`, `high`, and
  `max`, and explicit caller choices should survive unchanged.
- The catalog should expose the longest supported context but document that
  automatic routing can select a 262K provider. Enforce ZDR and re-check route
  policy and context limits before confidential or very long-context work.

## Implementation

1. Add an explicit `ModelSpec` with provider `openrouter`, backend
   `openai_compatible`, `local=False`, verified context/output/capability and
   disclosed parameter metadata, and unsupported-parameter filtering.
2. Add catalog, factory-routing, local-fallback, OpenRouter key, native-tool,
   structured-output, API discovery, and frontend selector coverage.
3. Document the exact route, independent evidence, provider claims, pricing,
   reliability, route-dependent context, and privacy guidance.
4. Run focused Ruff/format, mypy, pytest, affected frontend tests, native API
   smoke, Docker/log/curl smoke where feasible, and the Geist pre-push loop.
   Do not make a paid inference call.
