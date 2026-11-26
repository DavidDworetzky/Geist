# OpenRouter Claude Fable 5.2

## Objective

Add the stable `anthropic/claude-fable-5.2` OpenRouter route as an explicit
hosted Geist model on the existing OpenAI-compatible `OnlineAgent` path, with no
new dependency or agent architecture.

## Evidence and gate

Evidence checked November 10, 2026:

- OpenRouter live model API: https://openrouter.ai/api/v1/models
- OpenRouter model page: https://openrouter.ai/anthropic/claude-fable-5.2
- OpenRouter machine-readable model card:
  https://openrouter.ai/anthropic/claude-fable-5.2/llms.txt
- OpenRouter endpoint API:
  https://openrouter.ai/api/v1/models/anthropic/claude-fable-5.2-20261109/endpoints
- OpenRouter ZDR endpoint inventory: https://openrouter.ai/api/v1/endpoints/zdr
- OpenRouter provider policies: https://openrouter.ai/providers
- Anthropic launch announcement and provider evaluations:
  https://www.anthropic.com/claude-fable-and-mythos-5-2
- Anthropic system card:
  https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620/Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf
- Independent Artificial Analysis evaluation:
  https://artificialanalysis.ai/models/claude-fable-5-2
- Independent Vals evaluation:
  https://www.vals.ai/models/anthropic_claude-fable-5-2

Verified OpenRouter metadata: stable ID, 1,000,000-token context, 128,000-token
output, text/image/file input and text output, streaming, mandatory adaptive
reasoning with `high` as the default, automatic native tool calling, and
JSON-schema structured output. The standard price is $8/M input and $40/M
output, with $0.20/M cached input and separate cache-write rates. Google Vertex,
Amazon Bedrock, Azure, and Anthropic endpoints were live; the routed model page
reported 99.93% three-day availability. OpenRouter advertises `stop` but not
`n`, temperature, `top_p`, frequency penalty, or presence penalty.

Provider claims: Anthropic reports 49.1% FrontierCode 2.0 at xhigh, 69.9%
FrontierSWE v2, 74.6% Terminal-Bench-Science 0.1, and 74.2% Humanity's Last
Exam with tools. These results use Anthropic's evaluation harnesses and are not
treated as independent evidence.

Independent evidence: Artificial Analysis ranked max effort first with a 65
Intelligence Index, but measured $26.54 per task and a 235-second time to first
answer token. Vals ranked high effort first at 71.31% on its index and $17.94
per test, including 93.77% Vibe Code Bench and 65.15% Terminal-Bench 4.0, while
also reporting weak 5.00% HLAB-Vals. These results support a frontier option for
the hardest Geist/Pitchblend tasks, not a cost-efficient default.

Privacy and safety: OpenRouter's model page says ordinary Fable usage requires
retention for safety monitoring, and the exact model was absent from the live
ZDR endpoint inventory. The model is not recommended for confidential or
regulated workloads. No community report is required to clear the gate; early
community reports are mixed on cost and usage consumption and are supporting
evidence only.

Gate score: capability value 5, evidence quality 5, Geist fit 5, operational
safety 3, implementation confidence 4 = **22/25**. Operational safety is capped
because no ZDR-eligible route exists. Implementation confidence is capped
because this is a launch-day mandatory-reasoning route with a narrower generic
parameter contract than Geist's default payload.

## Implementation

1. Add a hosted `ModelSpec` with provider `openrouter`, backend
   `openai_compatible`, the exact stable ID and verified metadata, no parameter
   count, and `local=False`.
2. Filter unsupported generic defaults, apply mandatory high reasoning, and
   retain `stop`, automatic native tools, streaming, and structured output.
3. Extend catalog, factory-routing, API-discovery, credential, and request-shape
   coverage, and document pricing, evaluation evidence, and the non-ZDR caveat.

## Validation

1. Run focused Ruff lint/format, mypy, catalog and OnlineAgent pytest coverage,
   and the affected frontend model selector suite.
2. Run native model API discovery and hosted-only local-fallback smoke checks.
3. Apply the Geist pre-push loop in proportion to this catalog-only change,
   including Docker startup/log/curl and browser smoke when available.
4. Review the final diff, run staged secret checks, and make no paid inference
   request or credential-file inspection.
