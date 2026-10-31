# Meta Model API Muse Spark 1.3 integration plan

## Goal

Expose `muse-spark-1.3` through Meta's first-party Model API as a hosted Geist
model without adding dependencies or allowing local-loading fallback.

## Evidence

Evidence checked on 2026-09-03:

- Meta launch post: https://research.meta.ai/blog/introducing-muse-spark-1-3
- Meta Model API cookbook configuration:
  https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/README.md
- Meta Chat Completions recipe:
  https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/01_chat_completions.ipynb
- Meta reasoning recipe:
  https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/06_reasoning_tokens.ipynb
- Meta Muse Spark 1.1 launch:
  https://research.meta.ai/blog/introducing-muse-spark-meta-model-api
- Meta Muse Spark 1.2 availability:
  https://research.meta.ai/blog/multimodal-intelligence-of-muse-spark-1-2

Meta documents the model ID `muse-spark-1.3`, the OpenAI-compatible base URL
`https://api.meta.ai/v1`, the `MODEL_API_KEY` credential, a 1,048,576-token
context window, streaming, vision, function calling, and configurable
reasoning. The release is hosted-only; open weights are future work.

## Implementation

1. Add a direct `meta` provider backed by Meta Model API.
2. Add Muse Spark 1.1, 1.2, and 1.3 as direct-provider options with verified
   hosted capabilities and no guessed output limit or parameter count. Keep
   the Contributor tier on OpenRouter because direct API availability is not
   reliable enough to promise in the catalog. Conservatively omit optional
   request parameters not demonstrated by Meta's public cookbook.
3. Treat Meta Model API as a native-tool-capable OpenAI-compatible endpoint.
4. Expose the provider through the existing model API/UI path and document its
   credential.
5. Add focused provider, catalog, routing, key-resolution, and UI tests.

## Validation

- Focused Python catalog, factory, and OnlineAgent tests.
- Focused React provider/model visibility test.
- Formatting/lint checks for changed files.
- Proportional Geist test-loop checks; no live inference request without a
  configured Meta Model API credential.
