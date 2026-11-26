# OpenRouter Auto Code

## Goal

Add the stable `openrouter/auto-code` router as an explicit hosted model so
Geist and Pitchblend coding sessions can use OpenRouter's per-turn model and
reasoning selection through the existing OpenAI-compatible `OnlineAgent`.

## Scout gate

Research was refreshed on November 7, 2026 against OpenRouter's live model,
endpoint, model-page, router-documentation, benchmark, provider-policy, and ZDR
surfaces. The exact ID is absent from `origin/main` and every open pull request.

| Dimension | Score | Rationale |
| --- | ---: | --- |
| Capability value | 5 | Auto Code selects stronger coding models and reasoning efforts per turn, tracks failures, and reports performance in line with Claude Opus 5.5 at lower task cost. |
| Evidence quality | 3 | The request contract and five benchmark suites are documented, but the launch evaluation is OpenRouter-run and the new generation has no independent reproduction yet. |
| Geist fit | 5 | The stable ID exposes Chat Completions, streaming, native tools, structured output, and optional reasoning through Geist's existing OpenRouter path. |
| Operational safety | 4 | The router honors account/request governance and ZDR restrictions, but downstream model, provider, retention policy, and price vary unless administrators constrain them; there is no EU in-region route. |
| Implementation confidence | 4 | A catalog entry and `n` filtering are sufficient, but Geist does not yet expose Auto Router plugin controls or explicit session IDs. |
| **Total** | **21/25** | Clears the 20/25 gate with no score below 3. |

## Verified contract

- Stable OpenRouter ID and canonical slug: `openrouter/auto-code`.
- Published November 5, 2026; announced November 6, 2026.
- Provider/backend/local: `openrouter` / `openai_compatible` / `false`.
- Advertised combined context: 2,000,000 tokens.
- Advertised maximum output: 65,536 tokens.
- Input modalities: text, image, file, audio, and video; output modalities:
  text and image. Geist currently exposes only its supported text/image request
  surface and must not imply support for the other modalities or image output.
- Chat Completions, Completions, Messages, and Responses endpoints are listed.
- Streaming, optional reasoning controls, native tools/tool choice, parallel
  tool calls, response formats, and structured output are listed.
- `n` is not advertised; Geist's other generic sampling defaults are listed.
- Auto Code has no fixed per-token price. The caller pays the chosen model's
  standard price, and the response identifies that model.
- OpenRouter says Auto Code honors guardrails, ZDR requirements, provider
  restrictions, BYOK keys, and allowed/excluded model lists. The virtual
  router itself has no endpoint row in the ZDR inventory; confidential use
  therefore requires an account guardrail or per-request ZDR enforcement.
- The route is not available through OpenRouter's EU in-region endpoint.
- No parameter count or static underlying model family applies to this router.

## Evidence

Verified facts and provider documentation, checked November 7, 2026:

- https://openrouter.ai/api/v1/models
- https://openrouter.ai/openrouter/auto-code
- https://openrouter.ai/openrouter/auto-code/llms.txt
- https://openrouter.ai/docs/guides/routing/routers/auto-code
- https://openrouter.ai/docs/guides/routing/routers/auto-router
- https://openrouter.ai/docs/guides/features/zdr
- https://openrouter.ai/providers

Provider evaluation and claims:

- https://openrouter.ai/blog/announcements/auto-and-auto-code-routers/
- https://openrouter.ai/benchmarks/routers

OpenRouter reports normalized scores of 8.5 on Deep SWE, 8.7 on SWE Atlas QA,
7.6 on Terminal Bench 4.0, 8.9 on Tau3 Banking, and 9.7 on MMMU Pro Vision in
the November 5 benchmark run. It claims the aggregate is in line with Claude
Opus 5.5 at 74% lower cost per task, 4x fewer invalid tool calls than the prior
router during beta, and half as many cut-off agent responses. These are
provider-run results, not independent reproductions.

No credible independent evaluation of this exact November 2026 Auto Code
generation was available at review time. Community discussion usefully warns
that unconstrained automatic routing can make cost and behavior variable, but
it is supporting evidence only.

## Implementation

1. Add one explicit hosted `ModelSpec` for `openrouter/auto-code`, using the
   existing OpenRouter provider and `OPENROUTER_API_KEY` resolution.
2. Publish only metadata Geist can use: text/image input, native tools,
   optional reasoning, streaming, structured output, 2M context, and 65,536
   maximum output. Filter the unsupported generic `n` field.
3. Keep the entry hosted-only so factory inference cannot fall through to a
   local loader.
4. Add focused catalog, factory, API, OnlineAgent request/tool, and frontend
   discovery coverage.
5. Document variable pricing/model selection, provider-run evidence, ZDR and
   governance requirements, lack of EU routing, and the absence of explicit
   Auto Router plugin controls in Geist.

## Validation

Run change-scoped Ruff and format checks, targeted mypy, relevant catalog,
factory, API, and OnlineAgent pytest suites, the affected frontend Jest suite,
native API discovery and local-load rejection smokes, and the Geist pre-push
test loop. Attempt Docker startup/log/curl/browser checks if the daemon is
available. Do not make a paid inference call or inspect credential files.
