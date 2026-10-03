# OpenRouter Ling 3.1 Flash

## Goal

Add the stable hosted `inclusionai/ling-3.1-flash` model to Geist through the
existing OpenRouter/OpenAI-compatible path without introducing a new runtime or
allowing the 560B model to fall through to local loading.

## Verified contract (2026-10-02)

- OpenRouter model ID: `inclusionai/ling-3.1-flash`
- Provider/backend/local: `openrouter` / `openai_compatible` / `false`
- Served context/output limits: 262,144 / 32,768 tokens
- Modalities: text input and text output
- Parameters: 560B total and 25B activated, disclosed by inclusionAI in the
  OpenRouter metadata
- Reasoning: optional hybrid reasoning; enabled by default by the live route
- Capabilities: streaming and native tools/tool choice; no JSON-schema
  structured-output support in the current OpenRouter parameter contract
- Current OpenRouter pricing: free input and output on the sole Novita route
- Privacy: the endpoint is in OpenRouter's ZDR inventory; OpenRouter lists
  Novita as zero retention and no training
- Stability: named, non-preview ID, but the route is new, single-provider, and
  promotional. Re-check pricing and availability before production use.

## Scout gate

Capability value 4, evidence quality 3, Geist fit 4, operational safety 4,
implementation confidence 5: **20/25**. No dimension is below 3.

Provider benchmark claims are promising for agentic and coding work, but no
independent established leaderboard result exists yet. Independent hands-on
reports are mixed on long-horizon tasks. The material Geist value is a stable,
disclosed, zero-cost, ZDR-eligible tool-capable route for agent experiments,
not proven frontier quality.

## Implementation

1. Add an explicit hosted `ModelSpec` with the exact served limits,
   capabilities, disclosed parameter counts, and operational caveats.
2. Reuse the existing OpenRouter provider and `OPENROUTER_API_KEY` resolution.
3. Add catalog, local-loading rejection, factory-routing, API exposure, and
   OnlineAgent request-contract coverage while preserving native tool calls.
4. Add the model to user-facing provider documentation and the focused model
   selector regression test.

## Validation

- Targeted Ruff lint and format checks
- Targeted mypy for the catalog, OnlineAgent, factory, and registry
- Focused catalog/factory/API and OnlineAgent tests
- Affected frontend model-selector tests
- Native catalog API smoke
- Geist pre-push loop in proportion to the change, including Docker/log/curl
  and browser smoke if the local environment supports them
- Diff, secret, and code-quality review before commit and push

No paid inference call is required. Never read or print credential files or
credential values.
