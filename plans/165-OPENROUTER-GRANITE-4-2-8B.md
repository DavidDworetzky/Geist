# OpenRouter Granite 4.2 8B

## Goal

Expose IBM Granite 4.2 8B as an explicit hosted OpenRouter model while preserving
Geist's OpenAI-compatible routing, native tool calling, and local-load safety.

## Verified metadata

- Stable OpenRouter ID: `ibm-granite/granite-4.2-8b`
- Canonical route: `ibm-granite/granite-4.2-8b-20260831`
- OpenRouter addition: 2026-08-31T20:06:20Z
- Text input/output, 131,072-token context, 117,964-token maximum output
- Optional reasoning with full, low-effort, and non-thinking modes
- Native tools, tool choice, streaming, response format, and structured outputs
- OpenRouter price on 2026-09-01: $0.10/M input, $0.15/M output,
  $0.05/M cached input
- One CoreWeave BF16 endpoint, listed by OpenRouter as ZDR/no-training
- IBM-disclosed 8B dense model, released under Apache-2.0

## Gate

Score: 23/25 — capability value 4, evidence quality 5, Geist fit 5,
operational safety 4, implementation confidence 5. The single new upstream
route limits redundancy, but the stable model ID, independent Artificial
Analysis evaluation, native agent capabilities, and ZDR endpoint clear the
required gate.

## Plan

1. Add a hosted `ModelSpec` using the existing `openrouter` provider and
   `openai_compatible` backend; prevent local fallback and omit unsupported `n`.
2. Extend catalog, factory-routing, API exposure, key-resolution, and native-tool
   request tests for the exact model ID.
3. Document current limits, pricing, reasoning behavior, evidence, and the
   single-route/ZDR caveat.
4. Run focused Ruff/format, mypy, pytest, frontend catalog tests, native API
   smoke, and the Geist pre-push loop in proportion to the hosted-only change.

## Evidence

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/ibm-granite/granite-4.2-8b
- https://openrouter.ai/api/v1/models/ibm-granite/granite-4.2-8b-20260831/endpoints
- https://openrouter.ai/api/v1/endpoints/zdr
- https://openrouter.ai/docs/guides/features/zdr
- https://huggingface.co/ibm-granite/granite-4.2-8b
- https://www.ibm.com/granite
- https://artificialanalysis.ai/models/granite-4-2-8b
