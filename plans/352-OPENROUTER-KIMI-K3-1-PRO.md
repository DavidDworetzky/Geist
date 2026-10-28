# OpenRouter Kimi K3.1 Pro

## Goal

Add the stable `moonshotai/kimi-k3.1-pro` route to Geist's hosted model
catalog without introducing a new runtime or allowing the model to fall
through to local loading.

## Verified contract

- Provider/backend: OpenRouter through the existing OpenAI-compatible
  `OnlineAgent`.
- Context/output: 1,048,576 / 943,718 tokens in OpenRouter's live catalog.
- Input/output: text, image, and video input; text output.
- Capabilities: streaming, native tools, tool choice, and JSON-schema
  structured output.
- Reasoning: always enabled by the first-party Kimi API, with `max` as the
  default effort. Geist should apply `max` when the caller does not specify an
  effort.
- Request constraints: omit generic `n`, `temperature`, and `top_p`; the
  first-party API fixes these values and recommends that callers not send them.
- Privacy: both OpenRouter routes observed on 2026-10-24 were listed in the ZDR
  inventory. Confidential use should still enforce ZDR and re-check endpoint
  policy.

## Implementation

1. Add one explicit hosted `ModelSpec` with exact OpenRouter metadata and no
   guessed parameter count.
2. Add catalog, factory-routing, local-fallback, API-discovery, key-resolution,
   request-shaping, and native-tool coverage.
3. Add a focused frontend catalog-rendering test and operator documentation.
4. Run targeted Ruff/format, mypy, pytest, frontend tests, native API smoke,
   and the applicable Geist pre-push checks. Run Docker/log/curl/browser smoke
   when the local daemon permits it.

## Evidence and caveats

- OpenRouter added the stable route on 2026-10-23. The two current Moonshot
  service tiers were healthy when reviewed, but the route has less than one
  full day of operating history.
- Artificial Analysis independently measured a 55 Intelligence Index and a
  67% Terminal-Bench 4.0 result; LiveBench 2.0 reported 76.7 overall.
- Moonshot's additional coding, finance, 3D, and agent results remain provider
  claims. Community launch reports are supporting evidence only.
- OpenRouter's standard and high-speed endpoints are both ZDR-listed, and its
  provider directory lists Moonshot as zero retention/no training. ZDR should
  still be enforced for confidential or regulated traffic.
