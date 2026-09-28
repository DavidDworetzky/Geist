# OpenRouter Grok 4.7

## Objective

Add the stable `x-ai/grok-4.7` OpenRouter route as a hosted,
OpenAI-compatible Geist model without introducing a dependency or a new agent
architecture.

## Evidence and gate

Evidence checked September 28, 2026:

- OpenRouter live model API: https://openrouter.ai/api/v1/models
- OpenRouter endpoint API:
  https://openrouter.ai/api/v1/models/x-ai/grok-4.7/endpoints
- OpenRouter model page: https://openrouter.ai/x-ai/grok-4.7
- OpenRouter ZDR endpoint inventory: https://openrouter.ai/api/v1/endpoints/zdr
- OpenRouter provider policies: https://openrouter.ai/providers
- SpaceXAI model documentation: https://docs.x.ai/developers/grok-4-7
- SpaceXAI launch announcement: https://x.ai/news/grok-4-7
- Artificial Analysis model measurements:
  https://artificialanalysis.ai/models/grok-4-7

Verified facts: OpenRouter exposes the non-preview stable ID `x-ai/grok-4.7`
with a 500,000-token context window, a 450,000-token output ceiling, text/image/
file input, streaming, mandatory reasoning, native tools with forced function
selection, and JSON-schema structured output. The live route advertises
`low`, `medium`, `high`, and `xhigh` reasoning efforts with `high` as the
default. It lists $2/M input, $6/M output, and $0.50/M cached input, with
double token rates beyond 200,000 input tokens. Five xAI endpoints were live;
three appeared in OpenRouter's ZDR inventory. The standard ZDR endpoint showed
100% 30-minute and 97.95% one-day uptime at inspection time.

Provider claims: SpaceXAI reports 46.3% on CursorBench 4.0, 71.0% on DeepSWE
v1.1 at high effort, 37.6% on Terminal-Bench 4.0, and substantial improvements
over Grok 4.6. SpaceXAI also describes a new safeguard stack and says business
API data is not used for model training.

Independent evidence: Artificial Analysis reports an Intelligence Index of
46.4 and a Coding Agent Index of 56 for Grok Build with Grok 4.7 at xhigh,
up from 47 for Grok 4.6. OpenRouter's Design Arena data is more mixed, placing
the model 14th for game development and 50th for websites at inspection time.

Community reports are mixed: some practitioners report improved planning and
review, while others report greater verbosity and uneven frontend coding.
These reports support caveats but are not used as benchmark evidence.

Inference: Grok 4.7 materially improves Geist's existing Grok 4.6 option for
long-running coding and agent work at the same list price, while the independent
and arena evidence argues against making it the universal default.

Gate score: capability value 5, evidence quality 5, Geist fit 5, operational
safety 4, implementation confidence 5 = **24/25**. Operational safety is 4
because OpenRouter lists xAI's ordinary provider policy as 30-day retention;
confidential workloads must explicitly enforce a ZDR endpoint even though
xAI says business API data is not used for training.

## Implementation

1. Add an explicit hosted `ModelSpec` with provider `openrouter`, backend
   `openai_compatible`, the exact stable ID, verified limits/capabilities, no
   parameter-count claim, and no local fallback.
2. Preserve native tools and structured output, apply mandatory `high`
   reasoning, and filter unsupported `n`, frequency-penalty,
   presence-penalty, and stop parameters.
3. Extend catalog, factory-routing, credential, request-shaping, API, and
   frontend discovery coverage.
4. Document pricing, evidence limitations, and the requirement to enforce
   OpenRouter ZDR for confidential workloads.

## Validation

1. Run focused Ruff format/lint and mypy for changed Python paths.
2. Run focused catalog, factory-routing, OnlineAgent, API, and affected
   frontend model-selector tests.
3. Run the Geist pre-push test loop in proportion to this hosted catalog
   change, including native API and isolated Docker/browser smoke when
   available.
4. Review the final diff and staged secret scan. Do not make a paid inference
   request or inspect credential files.
