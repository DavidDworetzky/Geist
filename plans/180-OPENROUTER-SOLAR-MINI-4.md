# OpenRouter Solar Mini 4

## Goal

Add Solar Mini 4 as an explicit hosted OpenRouter model without changing
Geist's inference architecture or allowing the model to fall through to local
loading.

## Verified contract

- Model ID: `upstage/solar-mini4`
- Provider/backend: `openrouter` / `openai_compatible`
- Context/output limits: 524,288 / 131,072 tokens
- Inputs/outputs: text input and text output
- Capabilities: optional reasoning, streaming, native tools/tool choice,
  parallel tool calls, and JSON-schema structured output
- Request constraint: omit unsupported `n` and `stop`; retain supported
  sampling, reasoning, and tool parameters
- Disclosed size: 35B total parameters / 3B active parameters
- Privacy: OpenRouter lists one Upstage endpoint as ZDR and another as
  retaining prompts; confidential workloads must enforce ZDR routing

## Evidence and decision

Solar Mini 4 clears the daily scout gate at 21/25: capability value 4,
evidence quality 3, Geist fit 5, operational safety 4, and implementation
confidence 5. Its material benefit is inexpensive long-context, tool-capable
inference rather than proven frontier coding quality. OpenRouter reported
99.94% three-day availability and 99.97% 24-hour availability on September 27,
2026. Neither Upstage nor an established independent benchmark provider has
published an exact-model coding or agent score, so the model must not become a
default without Geist/Pitchblend qualification.

Sources checked September 27, 2026:

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/upstage/solar-mini4
- https://openrouter.ai/api/v1/models/upstage/solar-mini4-20260922/endpoints
- https://openrouter.ai/api/v1/endpoints/zdr
- https://openrouter.ai/providers/
- https://www.orcarouter.ai/blog/solar-mini-4-release

## Implementation

1. Add one hosted `ModelSpec` to `agents/model_catalog.py`, reusing the existing
   OpenRouter provider and credential resolution.
2. Cover metadata, local-load rejection, provider endpoint inference, API
   discovery, key resolution, native tool support, and request filtering.
3. Document selection, capabilities, pricing, evidence limits, reliability,
   and the need to enforce ZDR before confidential use.
4. Add a focused frontend catalog rendering test.

## Validation

Run focused Ruff/format, mypy, catalog and OnlineAgent pytest coverage, the
affected frontend test, native API discovery smoke, Docker/browser smoke when
available, and the Geist pre-push test loop. Do not make a paid inference call.
