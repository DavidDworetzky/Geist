# OpenRouter Particle-1

## Goal

Add the stable `particle-ai/particle-1` route to Geist's hosted model catalog
without changing the existing OpenAI-compatible architecture or adding a
dependency.

## Selection gate

Research refreshed on November 1, 2026 against OpenRouter's live catalog,
model and endpoint inventories, `origin/main` at `6b2d35b`, and all 30 open
pull requests.

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Capability value | 4 | Independent analytics testing beat the existing DeepSeek V4.1 Flash baseline by four tasks and tied GPT-6.1 Terra at materially lower cost; provider coding results show comparable success with shorter traces |
| Evidence quality | 4 | OpenRouter and Particle publish the contract and privacy policy; Plotly and Enclave provide independent exact-model task results, but no broad established leaderboard score is available yet |
| Geist fit | 5 | The stable OpenRouter ID exposes Chat Completions, streaming, native tools, optional reasoning, vision, and JSON output through the existing OnlineAgent |
| Operational safety | 4 | The sole route is ZDR-listed, no-training, healthy, and stable, but it is young, single-provider, and has no SLA |
| Implementation confidence | 5 | Exact limits and request parameters are live, and the model uses existing catalog-driven OpenRouter routing with only `n` requiring removal |
| **Total** | **22/25** | Clears the required 20/25 gate with no dimension below 3 |

No model was created after the prior-run cutoff. Particle-1 is the material
update: OpenRouter's list and endpoint APIs now agree on price, its route has a
full day of reported uptime, and Plotly published the first strong independent
exact-model result after the prior scout.

## Verified contract

- Stable model ID: `particle-ai/particle-1`
- Canonical slug: `particle-ai/particle-1-20261027`
- Provider/backend/local: `openrouter` / `openai_compatible` / `false`
- Context/output: 1,048,576 / 131,072 tokens
- Input/output: text and image input; text output
- Reasoning: optional; high by default, with none through max available
- Capabilities: Chat Completions, streaming, native tools and all tool-choice
  modes, validated JSON-object output, and best-effort JSON-schema guidance
- Unsupported generic request field: `n`
- OpenRouter price checked 2026-11-01: $0.05/M input, $0.50/M output, and
  $0.003/M cache reads
- Current route: one Particle AI endpoint, status `0`, with 100% reported
  30-minute and one-day uptime when checked
- No parameter count or undisclosed model-family attribution

## Privacy and evidence

### Verified facts

OpenRouter's exact endpoint appears in its ZDR inventory. Particle's terms and
documentation say standard synchronous inference does not persist raw prompts
or outputs and does not use them for training without affirmative opt-in.
OpenRouter ZDR must still be enforced for confidential workloads so routing
cannot silently move to a retaining endpoint if the provider pool changes.

- OpenRouter catalog (checked 2026-11-01):
  https://openrouter.ai/api/v1/models
- OpenRouter exact endpoint API (checked 2026-11-01):
  https://openrouter.ai/api/v1/models/particle-ai/particle-1-20261027/endpoints
- OpenRouter model page (checked 2026-11-01):
  https://openrouter.ai/particle-ai/particle-1
- OpenRouter ZDR inventory (checked 2026-11-01):
  https://openrouter.ai/api/v1/endpoints/zdr
- Particle API model contract and reasoning controls (checked 2026-11-01):
  https://docs.particle.ai/ and https://docs.particle.ai/reasoning
- Particle structured-output contract (checked 2026-11-01):
  https://docs.particle.ai/structured-outputs
- Particle data-retention documentation (checked 2026-11-01):
  https://docs.particle.ai/data-retention
- Particle terms, effective 2026-08-25 (checked 2026-11-01):
  https://particle.ai/terms
- Particle service status (checked 2026-11-01):
  https://status.particle.ai/

### Provider claims

Particle's October 28 launch evaluation reports 28.4%-52.2% fewer output
tokens and 34.4%-51.6% lower average task cost than DeepSeek V4.1 Flash across
Terminal-Bench 2.1, DeepSWE 1.1, ReactBench, and ADE Bench, with pass rates
inside the base model's confidence intervals. These are provider-run results,
not independent validation:

- https://particle.ai/blog/introducing-particle-1

### Independent evidence

- Plotly's October 31 exact-model DataAnalyticsBench run scored 40/43 versus
  36/43 for DeepSeek V4.1 Flash, tied GPT-6.1 Terra, cost $0.35, and used a
  median 16 seconds per question:
  https://plotly.com/blog/particle-1-plotly-data-analytics-bench/
- Enclave's October 29 direct-API security-agent run verified 4/11 vulnerable
  runs and 2/4 task types, with no patched controls verified. It tied DeepSeek
  on verified runs but solved one fewer task and cost $32.13 overall, so it is
  counterevidence against assuming universal cost or capability gains:
  https://enclave.ai/blog/particle-1-verifies-4-of-11-runs-exploits-cost-1-dollar

### Community reports

No community report was strong or controlled enough to affect the score. The
recommendation relies on the official contract and the independent Plotly and
Enclave task results.

### Inference

Particle-1 materially improves Geist/Pitchblend's low-cost analytics option and
can reduce long agent traces on coding/data tasks without a new runtime path.
Its limited independent coverage and single young route argue for keeping it
non-default and enforcing ZDR rather than treating it as a general frontier
replacement.

## Duplicate check

- `particle-ai/particle-1` is absent from `origin/main`.
- The exact ID and model name are absent from the titles, branches, bodies, and
  searchable patches of all 30 open pull requests.
- Existing model PRs cover distinct models and were skipped, including Kimi
  K3.1 Pro, MiniMax M3.1 Flash, Mercury 3, Gemini 4 Argon, Mistral Large 4,
  Ling 3.1 Flash, Solar Mini 4, GPT-6 Luna/Sol, GLM 5.3/FlashX, Pareto, Fugu
  Ultra v2, Qwen3.8 Max 0902, Muse Spark 1.3 Contributor, Granite 4.2 8B, and
  Kimi K3.

## Implementation

1. Add one explicit hosted `ModelSpec` with exact limits and only the verified
   unsupported `n` filter.
2. Reuse the existing OpenRouter `ProviderSpec`, `OPENROUTER_API_KEY`, and
   OpenAI-compatible `OnlineAgent`; preserve optional reasoning and native
   tools.
3. Cover catalog metadata, factory routing, local-load rejection, key
   resolution, request filtering, native tools, structured output, API/UI
   discovery, and documentation.
4. Keep the model non-default while the single route is operationally young.

## Validation

1. Run focused Ruff lint/format and mypy on changed Python.
2. Run focused catalog, factory, OnlineAgent, route, and model API tests.
3. Run the affected frontend model-picker test.
4. Exercise native model discovery through the FastAPI route.
5. Run the Geist pre-push loop: applicable pre-commit hooks, Docker startup,
   backend/frontend logs, `localhost:3000` curl, and browser UI smoke when the
   local environment permits them.
6. Review the final diff for correctness, security, privacy language, and
   accidental secret exposure. Do not make a paid inference call.

Native `MLX_BACKEND=1` validation is not applicable because the model is
hosted-only and cannot enter a local runner.
