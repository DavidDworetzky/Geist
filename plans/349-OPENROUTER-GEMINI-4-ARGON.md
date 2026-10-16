# OpenRouter Gemini 4 Argon

## Goal

Add Google's stable Gemini 4 Argon route to Geist as a hosted-only OpenRouter
model without changing the existing OpenAI-compatible agent architecture.

## Evidence and gate

Checked October 15, 2026:

- OpenRouter exposes the stable interactive ID `google/gemini-4-argon` with a
  1,048,576-token context window, a 65,536-token synchronous output limit,
  text/image/file/audio/video input, mandatory low/medium/high reasoning,
  native tools, JSON-schema structured output, and streaming.
- Google's model documentation labels `gemini-4-argon` stable and documents
  the same modalities, function calling, structured output, and medium default
  reasoning. Google's migration guide says sampling parameters are ignored.
- Vals AI reports top-tier exact-model coding and agent results, including
  91.91% Vibe Code Bench, 68.17% Code Migration, and 57.58% Terminal-Bench 4.0.
- The exact OpenRouter ID appears in the ZDR model inventory. Google Vertex is
  listed as zero retention/no training, while Google AI Studio retains data for
  55 days; confidential workloads must enforce ZDR routing.

Gate score: capability value 5, evidence quality 5, Geist fit 5, operational
safety 4, implementation confidence 5 = 24/25. No dimension is below 3.

## Implementation

1. Add one explicit `ModelSpec` using provider `openrouter`, backend
   `openai_compatible`, and `local=False`; do not add parameter-count or family
   claims not published by Google.
2. Encode verified context/output limits and capabilities, require medium
   reasoning by default, and remove unsupported/ignored generic sampling
   parameters while preserving native tools and structured output.
3. Cover catalog metadata, local-load rejection, factory routing, API/UI
   visibility, OpenRouter key discovery, and request shaping.
4. Document pricing, benchmark evidence, reliability, and the ZDR requirement.

## Validation

- Targeted Ruff format/lint and mypy for changed Python.
- Focused catalog, factory-routing, API serialization, and OnlineAgent tests.
- Affected frontend catalog-selection tests.
- Native model-list/detail API smoke.
- Staged pre-commit/security checks and final diff review.
- Docker/log/curl/browser smoke when the local daemon is available; native MLX
  is not applicable to this hosted-only model.

## Sources

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/google/gemini-4-argon
- https://openrouter.ai/api/v1/models/google/gemini-4-argon-20261013/endpoints
- https://openrouter.ai/api/v1/models?zdr=true
- https://openrouter.ai/providers
- https://openrouter.ai/docs/guides/features/zdr
- https://ai.google.dev/gemini-api/docs/models/gemini-4-argon
- https://ai.google.dev/gemini-api/docs/latest-model
- https://www.vals.ai/models/google_gemini-4-argon
