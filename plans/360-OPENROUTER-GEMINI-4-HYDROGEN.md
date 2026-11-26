# OpenRouter Gemini 4 Hydrogen

## Goal

Add the stable `google/gemini-4-hydrogen` OpenRouter route as a hosted-only
Geist model without changing the existing online-agent architecture.

## Evidence and gate

OpenRouter added the stable Chat Completions ID on November 12, 2026. Its live
model and endpoint APIs advertise a 1,048,576-token combined context, up to
943,718 output tokens on long-decode routes, text/image/file/audio/video input,
mandatory reasoning, streaming, native tools, all tool-choice modes, and
JSON-schema structured output. Geist currently exposes text, image, tools, and
structured output through its existing OpenAI-compatible message contract.

Independent evaluations support a material fast-agent and coding improvement:
Vals reports 68.49% overall, 54.04% on Terminal-Bench 4.0, and 87.8% on Vibe
Code Bench; Artificial Analysis reports an Intelligence Index of 54 at xhigh
effort and 51 at high effort; CursorBench 4.0 reports 54.3% at high effort. The
scout gate is 24/25: capability 5, evidence 5, Geist fit 5, operational safety
4, and implementation confidence 5.

OpenRouter lists Google Vertex as no-training/zero-retention, and the exact
model has Vertex endpoints in the ZDR inventory. Those records currently allow
safety retention, while default routing can select Google AI Studio, which
OpenRouter lists with 55-day retention. Confidential workloads must enforce
ZDR and re-check endpoint policy.

## Implementation

1. Add one explicit hosted `ModelSpec` with provider `openrouter`, backend
   `openai_compatible`, and no parameter-count claim.
2. Apply OpenRouter's mandatory medium reasoning effort and filter the generic
   `n`, frequency-penalty, and presence-penalty defaults that its live contract
   does not advertise. Preserve caller-selected reasoning, sampling, stop,
   native tools, tool choice, and structured output.
3. Extend catalog, local-fallback, factory-routing, API-discovery,
   credential-resolution, online request-shape, and frontend selector tests.
4. Document pricing, endpoint-dependent output ceilings, evaluation evidence,
   reliability, and privacy caveats.

## Validation

Run focused Ruff and format checks, mypy for the catalog/online/factory path,
focused backend tests, affected frontend model-selection tests, native API
catalog smoke, and the repository pre-push test loop. Do not make a paid model
call. Record any Docker, browser, or environment blocker explicitly.
