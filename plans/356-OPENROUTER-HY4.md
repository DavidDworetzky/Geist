# OpenRouter Tencent Hy4

## Goal

Add Tencent's stable OpenRouter model alongside the still-callable Hy4 preview,
keeping both hosted-only on Geist's existing OpenAI-compatible agent path.

## Gate

- Capability value: 4/5
- Evidence quality: 5/5
- Geist fit: 5/5
- Operational safety: 5/5
- Implementation confidence: 5/5
- Total: 24/25; no score is below 3

The stable release adds image input and materially stronger long-horizon agent
coverage over the preview. Independent Vals and Artificial Analysis results
show useful coding and professional-task capability, although it remains below
the leading frontier models and can be verbose or slow.

## Verified metadata

- OpenRouter model ID: `tencent/hy4`
- Canonical route: `tencent/hy4-20261104`
- Provider/backend/local: `openrouter` / `openai_compatible` / `false`
- Modalities: text and image input, text output
- Context/output limits: 1,048,576 / 65,536 tokens
- Capabilities: optional reasoning (high by default), streaming, automatic
  native tools, response format, and JSON-schema structured output
- Unsupported Geist defaults: `n`, `top_p`, `frequency_penalty`, and
  `presence_penalty`
- Privacy: the sole Tencent route is no-training, zero-retention, and present
  in OpenRouter's ZDR inventory; it has no EU in-region option
- Model size/family details: not copied from the preview because the stable
  release had not yet published weights or a system card

## Implementation

1. Add a stable `ModelSpec` with the exact ID and verified limits while keeping
   the preview entry available for existing saved settings.
2. Keep OpenRouter key resolution, native tools, and data-driven request
   filtering; add no provider or dependency.
3. Update catalog, local-load rejection, factory routing, request-contract,
   API/UI discovery, and frontend selector tests.
4. Update operator documentation with stable-model and ZDR caveats.

## Validation

1. Run focused Ruff formatting/linting and mypy for changed Python files.
2. Run catalog, factory-routing, OnlineAgent, API, and affected frontend tests.
3. Run native API discovery without making a paid model call.
4. Follow the Geist pre-push test loop for Docker/log/curl/browser evidence when
   available; native MLX is not applicable to a hosted-only catalog change.
5. Review the final diff and run staged security/secret checks before push.

## Sources checked November 6, 2026

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/tencent/hy4
- https://openrouter.ai/api/v1/models/tencent/hy4-20261104/endpoints
- https://openrouter.ai/api/v1/endpoints/zdr
- https://openrouter.ai/Providers
- https://openrouter.ai/docs/guides/features/zdr
- https://www.tencent.com/zh-hk/tencent-launches-hy4-joining-the-ranks-of-the-worlds-leading-open-source-ai-models/
- https://www.vals.ai/models/tencent_hy4
- https://artificialanalysis.ai/models/tencent-hy4
- https://acceleratedlogicai.com/blog/tencent-hy4-html-artifact-test-glass-crane-procedural-globe-wizard/
