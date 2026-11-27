# OpenRouter Ember-1

## Goal

Add Fireworks Ember-1 as an explicit hosted OpenRouter model without changing
Geist's inference architecture or allowing the model to fall through to local
loading.

## Verified contract

- Model ID: `fireworks/ember-1`
- Provider/backend: `openrouter` / `openai_compatible`
- Context/output limits: 1,048,576 / 943,718 tokens
- Inputs/outputs: text and image input, text output
- Capabilities: optional reasoning, streaming, native tools/tool choice, and
  JSON-schema structured output
- Request constraint: omit unsupported `n`; retain supported sampling and tool
  parameters
- Privacy: the sole Fireworks route is listed by OpenRouter as zero retention
  and not used for training as of September 24, 2026

Parameter counts and architecture details are intentionally omitted because
Fireworks does not disclose them for Ember-1.

## Implementation

1. Add one hosted `ModelSpec` to `agents/model_catalog.py`, reusing the existing
   OpenRouter provider and credential resolution.
2. Cover metadata, local-load rejection, provider endpoint inference, API
   discovery, key resolution, native tool support, and request filtering.
3. Document selection, capabilities, pricing, benchmark provenance, and the
   need to re-check privacy policy before confidential use.
4. Add a focused frontend catalog rendering test.

## Validation

Run focused Ruff/format, mypy, catalog and OnlineAgent pytest coverage, the
affected frontend test, native API discovery smoke, Docker/browser smoke when
available, and the Geist pre-push test loop. Do not make a paid inference call.
