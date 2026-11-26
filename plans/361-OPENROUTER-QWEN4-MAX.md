# OpenRouter Qwen4 Max

## Objective

Add the stable `qwen/qwen4-max` OpenRouter route as an explicit hosted-only
Geist model. Reuse the existing OpenAI-compatible `OnlineAgent`, preserve
native tool calling and caller-selected reasoning, expose verified metadata to
the API/UI, and prevent local-loader fallback.

Research was refreshed on November 16, 2026 against the prior scout cutoff of
November 15, 2026 at 16:00:32 UTC, `origin/main`, and all 38 open pull
requests. Neither Qwen4 Max nor Hy4 flash is present in the catalog or an open
PR. Twenty-five existing model-related PRs were treated as in-flight work and
not duplicated.

## Gate

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Capability value | 5 | Independent coding, finance, and agent evaluations show a material gain over Qwen 3.8 Max at lower token cost. |
| Evidence quality | 5 | Live OpenRouter contracts, Alibaba documentation, Vals, and Artificial Analysis agree on the core capability profile. |
| Geist fit | 4 | Text, reasoning, native tools, and structured output fit `OnlineAgent`; Geist does not yet expose the route's full video capability. |
| Operational safety | 3 | Alibaba says it does not train on business data without consent, but the sole route retains prompts for up to 30 days and is not ZDR-eligible. |
| Implementation confidence | 5 | The stable ID uses the existing OpenRouter provider/key path and only requires filtering unsupported `n`. |

**Total: 22/25; no dimension is below 3.**

`tencent/hy4-flash` scored 19/25: capability 4, evidence 2, Geist fit 4,
operational safety 5, and implementation confidence 4. It is deferred because
the exact model has only provider-run launch evaluations, leaving evidence
quality below the gate despite an attractive zero-retention route and price.

## Evidence

### Verified facts

- OpenRouter live model API, checked November 16, 2026: stable ID,
  `qwen/qwen4-max-20261115` canonical slug, creation time November 16, 2026 at
  02:18:46 UTC, 1,000,000-token context,
  text/image/video input, 131,072 maximum completion tokens, optional
  default-on reasoning (`high` default; `none`, `low`, `high`, and `xhigh`),
  streaming Chat Completions, tools/tool choice, and structured output:
  https://openrouter.ai/api/v1/models
- Exact endpoint API, checked November 16, 2026: one Alibaba route, $1.50/M
  input, $4.50/M output, $0.075/M cache read, $1.875/M cache write, 98.82%
  one-day availability, and every generic Geist request parameter except `n`:
  https://openrouter.ai/api/v1/models/qwen/qwen4-max-20261115/endpoints
- OpenRouter provider and ZDR inventories, checked November 16, 2026: Alibaba
  is no-training but prompt-retaining; the exact model has no ZDR endpoint:
  https://openrouter.ai/providers and
  https://openrouter.ai/api/v1/endpoints/zdr
- Alibaba Model Studio documentation, updated November 16, 2026: 1M context,
  131,072 non-thinking output tokens, thinking, function calling, and
  structured output:
  https://www.alibabacloud.com/help/en/model-studio/text-generation-model
- Vals Index v2.2, published November 15, 2026: 63.20% overall, 66.68%
  Finance Agent v2.1, 48.48% Terminal-Bench 4.0, and 86.67% Vibe Code Bench
  v1.1: https://www.vals.ai/home
- Artificial Analysis, checked November 16, 2026: Intelligence Index 43 and
  $3.27 average cost per index task:
  https://artificialanalysis.ai/models/qwen4-max
- Alibaba's data-retention policy, checked November 16, 2026: standard calls
  retain prompts and responses for no longer than 30 days, and business data
  is not used for training without separate consent:
  https://docs.modelstudio.console.alibabacloud.com/en/model-studio/data-retention-policy

### Provider claims

- Alibaba's November 16 launch report discloses 2.4T total and 71B activated
  parameters, claims 35% fewer tokens and about 60% lower cost per task than
  Qwen 3.8 Max on its Work360 evaluation, and positions Qwen4 Max for coding,
  agentic, visual, and professional work:
  https://www.alibabacloud.com/en/press-room/alibaba-unveils-qwen4-max

### Community reports

- Launch-day hands-on reports are mixed: some users report strong long-horizon
  coding and tool use, while others report overthinking and slow completion.
  These reports support operational caution but do not set the gate:
  https://www.reddit.com/r/Qwen_AI/comments/1xtk5bm/did_anyone_get_a_chance_to_test_qwen_4max_properly/

### Inference

- Qwen4 Max materially improves Geist's Qwen option because independent Vals
  results rise from Qwen 3.8 Max while the live route lowers token prices.
- The route is suitable for non-confidential coding, analysis, and agent work;
  it is not suitable for confidential or regulated workloads while OpenRouter
  exposes only the prompt-retaining Alibaba endpoint.
- Geist should preserve optional reasoning instead of forcing `high`, because
  OpenRouter supports disabling reasoning and honors explicit effort choices.

## Implementation

1. Add an explicit `ModelSpec` with provider `openrouter`, backend
   `openai_compatible`, `local=False`, verified context/output/capability and
   disclosed parameter metadata, and unsupported-parameter filtering.
2. Add catalog, factory-routing, local-fallback, OpenRouter key, native-tool,
   structured-output, and API/UI discovery coverage.
3. Document the exact route, evidence, pricing, reliability, and privacy
   caveat without claiming OpenRouter ZDR support.
4. Run focused Ruff/format, mypy, pytest, affected frontend tests, native API
   smoke, Docker/log/curl smoke where feasible, and the Geist pre-push loop.
   Do not make a paid inference call.
