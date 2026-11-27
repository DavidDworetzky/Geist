# OpenRouter DeepSeek V4.1 Flash 1017

## Goal

Refresh Geist's hosted DeepSeek V4.1 Flash catalog entry to the stable
`deepseek/deepseek-v4.1-flash-1017` snapshot without adding provider plumbing,
dependencies, or another local-loading path. Preserve the previous
`deepseek/deepseek-v4.1-flash` ID as a request/routing alias.

## Scout results reviewed on 2026-10-17

OpenRouter's live API grew from the saved 465-ID snapshot to 470 IDs. The new
interactive IDs were DeepSeek V4.1 Flash 1017, Reflection Beam 1.0, Elorian
Glimpse, and Xing4.0 29B A4B Free; `qwen/qwen3.7-plus:batch` is asynchronous
and does not fit Geist's interactive OnlineAgent. The exact IDs were absent
from `origin/main` and the titles, branches, bodies, and diffs of all 33 open
pull requests.

| Candidate | Capability | Evidence | Geist fit | Safety | Confidence | Total | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek V4.1 Flash 1017 | 5 | 4 | 5 | 5 | 5 | 24 | Selected |
| Reflection Beam 1.0 | 3 | 5 | 5 | 5 | 5 | 23 | Qualified; defer under one-PR limit |
| Elorian Glimpse | 4 | 3 | 3 | 5 | 4 | 19 | Research preview; narrow visual fit and no independent exact-model validation |
| Xing4.0 29B A4B Free | 3 | 3 | 5 | 3 | 5 | 19 | Single slow/rate-limited route with weak availability and provider-led evidence |
| Qwen3.7 Plus Batch | 3 | 3 | 1 | 4 | 1 | 12 | Batch-only contract is incompatible with OnlineAgent |

## Verified facts

- OpenRouter added the stable ID on 2026-10-17 with a 1,048,576-token context,
  393,216-token advertised output ceiling, text/image input, text output,
  optional reasoning (enabled by default at high effort), native tools,
  response-format JSON, streaming-compatible chat parameters, and route-level
  prices starting at $0.10/M input, $0.50/M output, and $0.002/M cache reads:
  https://openrouter.ai/api/v1/models
  https://openrouter.ai/deepseek/deepseek-v4.1-flash-1017
- The endpoint API listed six routes when checked: InferenceNet, Cloudflare,
  three DeepInfra service tiers, and DeepSeek. Five were present in the
  ZDR-filtered model inventory. Active routes reported 100% 30-minute uptime;
  the model page reported 99.48% routed 24-hour availability:
  https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash-20261017/endpoints
  https://openrouter.ai/api/v1/models?zdr=true
- OpenRouter's provider directory lists InferenceNet, Cloudflare, and DeepInfra
  as no-training/zero-retention. It lists DeepSeek as retaining prompts and
  using data for training, so default routing is not appropriate for
  confidential workloads:
  https://openrouter.ai/providers
- DeepSeek's exact model card identifies this as a minor update, recommends a
  one-million-token context and at least 256K output allowance, documents
  multi-turn tools/images/reasoning support, and publishes the weights under
  MIT. The catalog leaves parameter fields unset because DeepSeek describes a
  552B backbone with asymmetric 8B input/16B output activation while the weight
  repository reports 763B stored parameters:
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017

## Provider claims

DeepSeek reports improvements over the September snapshot on every published
row, including Terminal-Bench 4.0 from 31.2 to 42.3, DeepSWE v1.1 from 74.2 to
77.0, ProgramBench from 20.3 to 29.1, AutomationBench from 54.8 to 60.5, and
Agents' Last Exam from 31.8 to 35.4. The model card names the Claude Code and
Mini-SWE-V2 harnesses, but the 1017 results remain provider-run.

## Independent evidence and community reports

- Artificial Analysis evaluated the preceding V4.1 Flash release at
  Intelligence Index 39 (max effort), while Vals reported a 51.32 Vals Index.
  Neither had published a separate exact-1017 rerun at review time:
  https://artificialanalysis.ai/models/releases/deepseek-v4-1-flash
  https://www.vals.ai/benchmarks/vals_index
- Same-day hands-on/community discussion was positive about the coding and
  terminal gains but explicitly questioned vendor-benchmark transfer to daily
  use. It is supporting evidence only and is not treated as independent
  validation:
  https://www.reddit.com/r/DeepSeek/comments/1x84tv8/psa_deepseekv41flash1017_update_official/

## Inference

The exact snapshot materially improves Geist's low-cost coding, terminal,
automation, and multimodal agent option while preserving the existing
OpenAI-compatible route. It is the strongest candidate despite the lack of an
exact independent rerun because the delta is directly measured against the
already evaluated predecessor and the request contract is unchanged. Enforce
OpenRouter ZDR before confidential or regulated use so requests cannot reach
DeepSeek's retaining/training endpoint.

## Implementation

1. Replace the old V4.1 Flash catalog entry with an explicit hosted 1017
   `ModelSpec`; retain the old ID as an alias and keep parameters undisclosed.
2. Update catalog, local-fallback, factory-routing, API-visibility, key
   discovery, request-shaping, and documentation coverage.
3. Preserve optional reasoning, native tools, and supported sampling fields;
   continue filtering only unsupported `n`.

## Validation

Run targeted Ruff format/lint, mypy, focused catalog/factory/OnlineAgent/API
pytest, the affected frontend model-selection test, native FastAPI discovery,
applicable pre-push hooks, and Docker/log/curl/browser checks when the daemon is
available. Native MLX is not applicable to this hosted-only change. Do not make
a paid inference call or read credential files.
