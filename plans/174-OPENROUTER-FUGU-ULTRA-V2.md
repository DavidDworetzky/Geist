# OpenRouter Fugu Ultra v2

## Goal

Add Sakana Fugu Ultra v2 as an explicit hosted OpenRouter model without
changing Geist's provider architecture or allowing the route to fall through
to local model loading.

## Research snapshot (2026-09-11)

### Verified facts

- OpenRouter model ID: `sakana/fugu-ultra-v2`
- OpenRouter release timestamp: 2026-09-11 05:43:03 UTC
- Hosted provider: Sakana AI through OpenRouter's existing OpenAI-compatible
  endpoint
- Context: 1,000,000 tokens
- Maximum completion: 128,000 tokens
- Input/output: text, image, and file input to text output
- Price: $5 per million input tokens, $30 per million output tokens, and $0.50
  per million cache-read tokens. Prompts above 272,000 tokens cost $10 per
  million input, $45 per million output, and $1 per million cache-read tokens.
- Request contract: mandatory reasoning with `high`, `xhigh`, and `max`
  efforts; OpenRouter defaults to `xhigh`; native `tools`, automatic
  `tool_choice`, structured outputs, built-in web search, and streaming are
  supported. The route does not advertise Geist's standard `max_tokens`, `n`,
  sampling, penalty, or stop fields.
- OpenRouter reported one Sakana endpoint, 100% routed uptime and about 90%
  successful availability over the first observed day, with a 69.6-second
  median latency and 68 output tokens/second.
- OpenRouter does not retain prompt/response content unless logging is enabled.
  Its provider table marks Sakana as not training on prompts routed through
  OpenRouter, but as retaining prompts. The retention duration is not
  specified, and the endpoint is not in OpenRouter's ZDR inventory.

Sources:

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/sakana/fugu-ultra-v2
- https://openrouter.ai/api/v1/models/sakana/fugu-ultra-v2-20260911/endpoints
- https://openrouter.ai/providers/
- https://openrouter.ai/docs/guides/features/zdr
- https://openrouter.ai/docs/faq
- https://console.sakana.ai/privacy-policy
- https://console.sakana.ai/terms-of-service

### Provider claims

Sakana's September 11 launch post reports that Fugu Ultra v2 is best or tied
for best on five of eight evaluated benchmarks, including 48.3 on Chartography
and 74.3 on DeepSWE. Sakana says the fixed worker pool excludes Fable 5,
Fable 5.1, and GPT-6 Astra. These are launch-day provider-run results, not
independent reproductions.

Sources:

- https://sakana.ai/fugu-max-release/
- https://arxiv.org/abs/2606.21228

### Community reports and inference

No credible model-specific hands-on review or independent benchmark run for
the September 11 v2 release was available at review time. Older Fugu Ultra
reports support the general orchestration design but do not validate v2.
It is therefore an inference, not a verified fact, that v2 will preserve its
launch scores in Geist's own agent loop.

## Gate

- Capability value: 5/5
- Evidence quality: 3/5
- Geist fit: 5/5
- Operational safety: 3/5
- Implementation confidence: 5/5
- Total: 21/25

The route clears the 20/25 gate with no score below 3. It materially adds a
high-capability multi-agent orchestrator behind the already-supported
OpenRouter wire protocol. The main caveats are launch-day evidence, slow first
response, early availability, a single upstream, fixed/opaque worker routing,
and non-ZDR prompt retention. It must not be used for confidential or regulated
workloads.

## Implementation

1. Add a hosted `ModelSpec` with exact OpenRouter metadata, no parameter-count
   claims, mandatory `xhigh` reasoning, and all unsupported Geist generation
   fields removed before dispatch.
2. Verify factory routing uses the existing OpenRouter provider, native tool
   calling remains enabled, and local creation is rejected.
3. Add catalog, request-contract, discovery, and documentation coverage.
4. Run focused formatting, lint, type, backend, frontend, API smoke, and the
   applicable Geist test loop before publishing.
