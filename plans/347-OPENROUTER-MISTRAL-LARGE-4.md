# OpenRouter Mistral Large 4

## Goal

Add the named OpenRouter public-preview route `mistralai/mistral-large-4-0`
as an explicit hosted Geist model without changing the existing
OpenAI-compatible architecture or allowing the trillion-parameter model to
fall through to local loading.

## Scout decision

The model scored 21/25 on October 6, 2026:

- capability value: 5/5
- evidence quality: 3/5
- Geist fit: 5/5
- operational safety: 3/5
- implementation confidence: 5/5

OpenRouter exposes one callable Mistral endpoint with a stable named model ID,
native tools, forced or automatic tool choice, JSON-schema structured output,
streaming, text/image input, a 524,288-token context window, and a 262,144-token
output ceiling. The route is compatible with `OnlineAgent` and materially adds
a strong finance/legal/coding evaluation option.

The deductions are deliberate. Mistral labels the release a public preview,
says training is still in flight, and has not yet published the weights. The
exact-model benchmark results are credible but currently appear through
Mistral's launch materials rather than public independent leaderboards. The
sole Mistral route retains requests for 30 days, reports no training, and is
absent from OpenRouter's ZDR-filtered model API. It must not be recommended for
confidential or regulated workloads.

## Verified metadata

- OpenRouter ID: `mistralai/mistral-large-4-0`
- canonical slug: `mistralai/mistral-large-4-0-20261006`
- OpenRouter addition: `2026-10-06T13:42:02Z`
- provider/backend/local: `openrouter` / `openai_compatible` / `false`
- modality: text and image input, text output
- context/output: 524,288 / 262,144 tokens
- disclosed architecture: 1.05T total / 49B active parameters, 1.6B vision
  encoder
- reasoning: no separate OpenRouter reasoning parameter; Mistral describes the
  preview as hybrid instruct-and-reasoning
- capabilities: streaming, native tools, all advertised tool-choice modes,
  response format, and structured output
- pricing checked October 6: $0.68/M input, $2.09/M output, $0.07/M cached input
- route inventory: one Mistral endpoint; 99.54% 30-minute and 99.64% one-day
  endpoint uptime at review time

## Evidence

Verified facts:

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/mistralai/mistral-large-4-0
- https://openrouter.ai/api/v1/models/mistralai/mistral-large-4-0-20261006/endpoints
- https://openrouter.ai/providers
- https://docs.mistral.ai/models/mistral-large-4-0

Provider-published benchmark evidence:

- https://mistral.ai/news/mistral-large-4/
- Mistral's launch charts report 62 on Artificial Analysis DeepSWE 1.1, 28 on
  Terminal-Bench 4, 54.7 on Vals AI Finance Agent v2, and 15.8 on Vals AI's
  Harvey Legal Agent Benchmark. The Artificial Analysis figures were privately
  evaluated ahead of the harness's public launch; these are not yet public
  leaderboard results.

Community evidence:

- Same-day community discussion is launch commentary rather than repeatable
  hands-on evaluation and is not used to raise the score.

Inference:

- The exact-model results support evaluation for coding and professional-agent
  workloads, but preview mutability and missing public independent validation
  make it unsuitable as a default.

## Implementation

1. Add one explicit hosted `ModelSpec` with the verified ID, limits,
   capabilities, disclosed parameter counts, and preview/privacy warning.
2. Keep all advertised sampling parameters and native tools; omit only Geist's
   generic `n`, which the route does not advertise.
3. Add focused catalog, local-rejection, factory-routing, API exposure,
   OnlineAgent key/capability, and request-contract coverage.
4. Add a frontend catalog rendering regression and document setup, evidence,
   pricing, preview status, and privacy limits.

## Validation

- targeted Ruff lint and format check
- targeted mypy
- focused catalog/factory/API/OnlineAgent pytest
- affected frontend Jest suite
- native catalog and `/api/v1/models/` smoke
- Docker startup, logs, localhost curl, and browser smoke when available
- change-scoped pre-commit checks and `git diff --check`
- final diff and security review

No paid inference call is required. Never read or print credential files or key
values.
