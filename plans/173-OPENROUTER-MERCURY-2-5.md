# OpenRouter Mercury 2.5 integration plan

## Goal

Expose the stable OpenRouter model ID `inception/mercury-2.5` as a hosted Geist
model without a new dependency, model-specific runner, or local-loading path.

## Verified metadata (2026-09-09)

- Provider/backend: `openrouter` / `openai_compatible`
- Lifecycle: stable production route released September 8, 2026
- Modalities: text input and text output
- Limits: 260,000-token context and 65,536-token output
- Capabilities: optional reasoning, streaming, native and parallel tool calls,
  and JSON-schema structured output
- Reasoning contract: `none`, `low`, `medium`, and `high`; default `medium`
- Unsupported Geist defaults: `n`, `top_p`, `frequency_penalty`, and
  `presence_penalty`
- Current OpenRouter launch pricing: $0.04/M input, $0.15/M output, and
  $0.004/M cached input; Inception lists undiscounted prices of $0.20/M,
  $0.75/M, and $0.02/M
- Operations: one Inception endpoint; 98.77% routed three-day availability and
  99.995% endpoint uptime over the latest day when inspected
- Privacy: the endpoint is in OpenRouter's live ZDR inventory and is marked as
  no-training; OpenRouter prompt/response content logging is opt-in

Inception reports 1,107 tokens/second on broadly available NVIDIA GPUs and a
40% intelligence gain over Mercury 2. OpenRouter's live model page showed a
lower observed p50 of 137 tokens/second and 0.67-second latency. A third-party
40-trial FeatBench harness study of the preview route produced 28/40 verified
resolutions across four coding harnesses, but it evaluated harness differences,
not Mercury against other models, and predates the stable route. These claims
are documented with their provenance rather than treated as equivalent tests.

## Gate

- Capability value: 4/5
- Evidence quality: 4/5
- Geist fit: 5/5
- Operational safety: 5/5
- Implementation confidence: 5/5
- Total: 23/25; no dimension below 3

Mercury materially improves Geist's latency/cost option for repeated agent,
subagent, routing, compaction, and tool-search calls while preserving the
existing OpenAI-compatible request and native-tool path.

## Implementation

1. Add one explicit hosted `ModelSpec` using the existing OpenRouter provider.
2. Omit unsupported sampling defaults while preserving optional reasoning,
   temperature, stop, structured output, and native tool payloads.
3. Cover exact metadata, API/UI discovery, factory routing, local-load
   rejection, OpenRouter key resolution, and request constraints.
4. Document selection, temporary launch pricing, single-endpoint reliability,
   evidence limits, and ZDR guidance.

## Validation

1. Run Ruff lint/format checks and mypy for changed Python files.
2. Run focused model-catalog, OnlineAgent, route, and frontend tests.
3. Run a native model API smoke without making a paid inference call.
4. Follow the Geist test-loop skill in proportion to this catalog-only change,
   reporting Docker, browser, or native checks that are blocked or inapplicable.
5. Review the staged diff for correctness, security, and performance.

## Sources

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/inception/mercury-2.5
- https://openrouter.ai/api/v1/models/inception/mercury-2.5/endpoints
- https://openrouter.ai/api/v1/endpoints/zdr
- https://openrouter.ai/providers
- https://openrouter.ai/docs/guides/privacy/provider-logging
- https://openrouter.ai/docs/guides/features/zdr
- https://www.inceptionlabs.ai/blog/introducing-mercury-2-5
- https://www.inceptionlabs.ai/models
- https://benchlm.ai/models/mercury-2-5
- https://www.critique.sh/blog/mercury-harness-study-v1
