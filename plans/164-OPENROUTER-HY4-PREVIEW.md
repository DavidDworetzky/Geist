# OpenRouter Hy4 Preview

## Goal

Add Tencent Hy4 Preview as an explicit hosted OpenRouter model without allowing
the 770B model to fall through to local loading or changing Geist's inference
architecture.

## Verified metadata

- OpenRouter model ID: `tencent/hy4-preview`
- Provider/backend: `openrouter` / `openai_compatible`
- Modalities: text input and text output
- Context/output limits: 1,048,576 / 64,000 tokens
- Parameters: 770B total, 49B activated per token
- Capabilities: optional reasoning, streaming, native tools, and structured output
- Unsupported Geist defaults: `n`, `top_p`, `frequency_penalty`, and
  `presence_penalty`
- Pricing on August 28, 2026: $0.834/M input, $2.501/M output, and $0.042/M
  cached input tokens
- Privacy: the sole Tencent endpoint is listed by OpenRouter as zero retention
  and not used for training; it remains a new single-provider preview route

## Implementation

1. Add one hosted `ModelSpec` using the existing OpenRouter provider.
2. Cover exact metadata, OpenRouter factory routing, local-load rejection,
   API/UI discovery, API-key resolution, native tool support, and unsupported
   request parameter removal.
3. Document selection, preview/reliability caveats, privacy, and pricing.

## Validation

1. Run Ruff lint/format checks and mypy for changed Python files.
2. Run focused model-catalog, OnlineAgent, and frontend component tests.
3. Run native model API readiness smoke without making an inference call.
4. Follow the Geist pre-push test loop, including Docker/log/curl and browser
   checks when the local environment supports them; report every blocked check.
5. Review the staged diff and run the repository security check before push.

## Sources

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/api/v1/models/tencent/hy4-preview-20260827/endpoints
- https://openrouter.ai/api/v1/endpoints/zdr
- https://openrouter.ai/providers
- https://openrouter.ai/docs/guides/features/zdr
- https://www.tencent.com/tencent-releases-and-open-sources-tencent-hy4-preview/
- https://huggingface.co/tencent/Hy4-preview
