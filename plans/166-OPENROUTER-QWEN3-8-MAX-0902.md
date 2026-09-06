# OpenRouter Qwen3.8 Max 0902

## Goal

Replace Geist's retired `qwen/qwen3.8-max` OpenRouter catalog entry with the
currently callable `qwen/qwen3.8-max-0902` snapshot and encode its verified
request contract and privacy caveats.

## Evidence and gate

- OpenRouter's live catalog removed `qwen/qwen3.8-max` and exposes
  `qwen/qwen3.8-max-0902` (permaslug `qwen/qwen3.8-max-20260902`).
- The route is a stable Alibaba endpoint with a 1,000,000-token context,
  131,072-token output limit, text/image/video input, streaming, native tools,
  structured output, and mandatory reasoning with `xhigh` as the default.
- OpenRouter lists $2/M input and $6/M output pricing. Alibaba does not train on
  requests, but retains prompts for an unspecified period and the endpoint is
  absent from OpenRouter's ZDR list.
- Gate score: 21/25 (capability 4, evidence 4, Geist fit 5, operational safety
  3, implementation confidence 5). No category is below 3.

## Implementation

1. Update the hosted `ModelSpec` to the exact live OpenRouter ID and remove
   unverified parameter-count metadata.
2. Encode the unsupported `n` parameter and mandatory `xhigh` reasoning effort.
3. Update focused catalog/factory/API and OnlineAgent request-contract tests.
4. Document the route replacement, metadata, price, and confidentiality caveat.

## Validation

Run targeted Ruff/format, mypy, backend catalog and OnlineAgent tests, affected
frontend tests, a native model-API smoke, and the Geist test loop where feasible.
Do not make a paid inference call.
