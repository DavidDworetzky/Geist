# OpenRouter Claude Haiku 5.5

## Goal

Add Anthropic Claude Haiku 5.5 as an explicit hosted OpenRouter model without
changing Geist's inference architecture or adding dependencies.

## Verified model contract

- OpenRouter model ID: `anthropic/claude-haiku-5.5`
- Provider/backend: `openrouter` / `openai_compatible`
- Hosted only (`local=False`)
- Context window: 1,000,000 tokens
- Maximum output: 128,000 tokens
- Inputs: text, image, and file; output: text
- Optional adaptive reasoning with low through max effort levels
- Native tool calling, streaming, and JSON-schema structured output
- Unsupported generic request fields: `n`, `temperature`, `top_p`,
  `frequency_penalty`, and `presence_penalty`
- No disclosed parameter count or model-family architecture attribution

## Implementation

1. Add a `ModelSpec` beside the other Claude 5.5 OpenRouter entries.
2. Reuse the existing OpenRouter provider, credential lookup, hosted-only
   routing, request filtering, and native tool-calling paths.
3. Extend catalog, factory-routing, OnlineAgent, API/UI, and documentation
   coverage for the exact model ID and verified metadata.
4. Document tiered pricing and the privacy caveat: the model has ZDR-capable
   routes, while default routing may also use Anthropic or Claude Platform on
   AWS endpoints that OpenRouter lists with 30-day retention.

## Validation

- Run targeted Ruff format/lint and mypy checks.
- Run focused catalog, factory, OnlineAgent, and API tests.
- Run affected frontend tests for live catalog rendering.
- Exercise native API model-list/detail smoke checks.
- Run the applicable Geist pre-push checks, Docker/log/curl smoke, and browser
  catalog smoke when the local environment permits.
- Do not make a paid model inference call.

## Risks and rollback

- OpenRouter metadata and endpoint policies can change; retain an explicit ZDR
  warning and re-check the live endpoint inventory before confidential use.
- Optional reasoning should remain provider-controlled rather than being made
  mandatory by Geist.
- Rollback is the removal of the single catalog entry and its focused tests and
  documentation.
