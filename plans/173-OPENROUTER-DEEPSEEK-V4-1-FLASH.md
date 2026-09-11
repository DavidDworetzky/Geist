# OpenRouter DeepSeek V4.1 Flash

## Goal

Add the stable `deepseek/deepseek-v4.1-flash` OpenRouter route as an explicit
hosted Geist model without changing provider plumbing or local-model loading.

## Evidence reviewed on 2026-09-10

### Verified facts

- OpenRouter's live model API lists the stable ID, release timestamp
  `1789021285`, 1,048,576-token context, 384,000-token advertised output
  ceiling, text/image input, text output, optional reasoning (default `high`),
  tools, response-format JSON, and streaming-compatible chat parameters:
  https://openrouter.ai/api/v1/models
- The OpenRouter model page lists four upstreams, 99.92% routed three-day
  availability, and current off-peak pricing of $0.15/M input, $0.60/M output,
  and $0.003/M cached input. Peak rates are twice those prices during the
  configured weekday windows:
  https://openrouter.ai/deepseek/deepseek-v4.1-flash
- OpenRouter exposes DeepInfra, Novita, and Venice instances of this route in
  its live ZDR inventory. Its provider table says the first-party DeepSeek
  endpoint retains prompts and may train on them:
  https://openrouter.ai/api/v1/endpoints/zdr
  https://openrouter.ai/providers/
- DeepSeek's release notes and model card confirm a 2026-09-10 stable release,
  native image understanding, one-million-token context, controllable
  reasoning, tool calling, and an MIT-licensed open-weight release:
  https://api-docs.deepseek.com/news/news260910/
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

The catalog intentionally leaves parameter fields unset. DeepSeek describes a
552B-parameter backbone with 8B active during prefill and 16B during decode,
while the Hugging Face repository reports 763B stored parameters and the
technical report separately describes a 196B Engram memory. A single catalog
`parameter_count` value would hide those different meanings.

### Provider claims

DeepSeek reports 90.6 on Terminal-Bench 2.1, 74.2 on DeepSWE v1.1, 64.0 on
NL2Repo-Bench, 88.1 on CyberGym, and 54.8 on AutomationBench at maximum
reasoning effort. The model card includes harness settings and reproduction
instructions, but these results are provider-run rather than independent.

### Community reports

Early reports are mixed and supporting-only. One same-prompt app build reported
38% lower elapsed time and 43% fewer total tokens than V4 Flash Vision Exp but
also left tasks incomplete. Another small defect suite tied the prior model and
missed one independently checked edge case. A separate hands-on report praised
speed but found an infrastructure-diagnosis failure:
https://www.reddit.com/r/DeepSeek/comments/1watjcl/deepseek_v41_flash_vs_v4_flash_vision_exp_38/
https://www.reddit.com/r/hermesagent/comments/1wcl6yj/ran_deepseek_v41_flash_as_my_hermes_delegation/
https://www.reddit.com/r/DeepSeek/comments/1wcem68/v41_i_tried_and_it_is_very_fast_but_reasoning/

### Inference

The route materially improves Geist's low-cost multimodal agent option, but
formal evidence remains launch-day and provider-led. Recommend it for
cost-sensitive agentic/coding evaluation, not as an unreviewed replacement for
the strongest catalog models. Confidential workloads must enforce OpenRouter
ZDR so they cannot reach the retaining/training first-party endpoint.

## Gate

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Capability value | 5 | Large gains claimed on agentic coding, native vision, 1M context |
| Evidence quality | 3 | Detailed/reproducible provider evidence; no independent benchmark yet |
| Geist fit | 5 | Chat Completions, tools, reasoning, and vision match OnlineAgent |
| Operational safety | 4 | Four upstreams and three ZDR choices; default route can retain/train |
| Implementation confidence | 5 | Existing OpenRouter provider and request-constraint machinery suffice |
| **Total** | **22/25** | Clears 20/25 with no dimension below 3 |

The exact ID is absent from `origin/main` and all 15 open pull requests checked
on 2026-09-10.

## Implementation

1. Add one hosted `ModelSpec` using the existing OpenRouter provider.
2. Mark `n` unsupported, preserve native tools, and leave reasoning optional.
3. Add catalog, local-fallthrough, factory-routing, key-resolution, request,
   and API-discovery coverage.
4. Document pricing, capability, evidence, provider variance, and privacy.

## Validation

Run targeted Ruff/format, mypy, catalog/OnlineAgent/API tests, affected frontend
model tests, applicable pre-commit checks, a native loopback API discovery
smoke, and the Geist test loop in proportion to this catalog-only change. Do
not make a paid inference call.
