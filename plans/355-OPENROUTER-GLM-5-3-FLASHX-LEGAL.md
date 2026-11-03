# OpenRouter GLM 5.3 FlashX Legal

## Goal

Expose Z.ai's stable `z-ai/glm-5.3-flashx-legal` route through Geist's existing
OpenRouter/OpenAI-compatible path without adding a dependency or allowing the
hosted model to fall through to local loading.

## Verified contract (2026-11-02)

- OpenRouter model page: <https://openrouter.ai/z-ai/glm-5.3-flashx-legal>
- Endpoint metadata: <https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flashx-legal-20261102/endpoints>
- Z.ai launch post (2026-10-12): <https://z.ai/blog/glm-legal>
- Independent Vals model card (2026-10-12 evaluation): <https://www.vals.ai/models/zai_glm-legal>
- Model ID: `z-ai/glm-5.3-flashx-legal`
- Context/output: 1,048,576 / 131,072 tokens
- Modalities: text, image, and video input; text output
- Features: mandatory reasoning (default `max`), streaming, native tools with
  automatic tool selection, and JSON-schema structured output
- Current OpenRouter price: $0.37/M input, $1.25/M output, and $0.075/M cached
  input
- Current route: one Z.AI FP8 endpoint, zero-retention/no-training and listed
  in OpenRouter's ZDR inventory; it is not available through EU in-region
  routing
- OpenRouter does not advertise `n`, `frequency_penalty`, `presence_penalty`,
  or `stop` for this route

Do not record a parameter count or activated-parameter count. The hosted model
metadata does not disclose those values for this legal specialization.

## Implementation

1. Add one explicit hosted `ModelSpec` using provider `openrouter`, backend
   `openai_compatible`, `local=False`, and the verified request constraints.
2. Keep OpenRouter's existing `ProviderSpec`; no duplicate provider or new
   dependency is needed.
3. Cover catalog metadata, factory routing/local rejection, OpenRouter key
   resolution, mandatory reasoning, unsupported-parameter filtering, native
   tool preservation, and API/UI model discovery.
4. Document the legal specialization, independent evidence, current price,
   single-route reliability, ZDR policy, lack of EU in-region routing, and the
   need for professional review in high-stakes legal work.

## Validation

- Ruff lint and format checks for changed Python files
- Targeted mypy for changed Python files
- Focused catalog, factory-routing, OnlineAgent, native-tool, and API tests
- Focused frontend model-selector tests
- Native FastAPI model-discovery smoke
- Docker startup/log/curl smoke when Docker is available
- Staged Geist pre-push hooks and final security/performance/correctness review

No paid inference call is required. Confirm credentials only by variable name
or presence; never read or print credential values.
