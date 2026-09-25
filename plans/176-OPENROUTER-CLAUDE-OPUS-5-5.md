# OpenRouter Claude Opus 5.5

## Objective

Add the stable `anthropic/claude-opus-5.5` OpenRouter route as a hosted,
OpenAI-compatible Geist model without introducing a dependency or a new agent
architecture.

## Evidence and gate

Evidence checked September 22, 2026:

- OpenRouter live model API: https://openrouter.ai/api/v1/models
- OpenRouter ZDR endpoint inventory: https://openrouter.ai/api/v1/endpoints/zdr
- OpenRouter provider policies: https://openrouter.ai/providers
- Anthropic model documentation:
  https://platform.claude.com/docs/en/models/opus-5-5/overview
- Anthropic launch announcement: https://www.anthropic.com/claude-opus-5-5
- Anthropic migration guide:
  https://platform.claude.com/docs/en/models/opus-5-5/migration-guide

Verified OpenRouter metadata: 1,000,000-token context, 128,000-token output,
text/image input, streaming, mandatory reasoning with `high` as OpenRouter's
default effort, native tools with automatic tool choice, and JSON-schema
structured output. OpenRouter lists $4/M input, $20/M output, and $0.20/M
cached input for the standard Bedrock route. Three Amazon Bedrock endpoints
appear in OpenRouter's ZDR inventory; the global route reported 99.80% one-day
uptime at review time. The aggregate route advertises temperature through
Azure, while the recommended ZDR endpoints omit it, so Geist filters
temperature to keep requests compatible with ZDR routing.

Gate score: capability value 5, evidence quality 5, Geist fit 5, operational
safety 5, implementation confidence 4 = **24/25**. Implementation confidence
is not 5 because this is a launch-day route with always-on reasoning and a
narrower sampling contract than Geist's generic OpenAI-compatible payload.

## Implementation

1. Add an explicit hosted `ModelSpec` with provider `openrouter`, backend
   `openai_compatible`, the exact stable ID, verified limits/capabilities, no
   parameter-count claim, and no local fallback.
2. Filter unsupported generic request parameters (`n`, temperature, `top_p`,
   frequency penalty, and presence penalty), apply mandatory high reasoning,
   and retain stop, response format, streaming, and native tools.
3. Extend catalog, factory-routing, credential, request-shaping, and API/UI
   discovery tests.
4. Document pricing, launch-day evidence, forced-tool caveats, and the need to
   enforce OpenRouter ZDR for confidential workloads.

## Validation

1. Run focused Ruff format/lint and mypy for the changed Python paths.
2. Run focused catalog, factory-routing, OnlineAgent, API, and affected
   frontend model-selector tests.
3. Run the Geist pre-push test loop in proportion to the catalog/runtime
   change, including native API smoke and isolated Docker startup/curl when
   available.
4. Review the final diff and staged secret scan. Do not make a paid inference
   request or inspect credential files.
