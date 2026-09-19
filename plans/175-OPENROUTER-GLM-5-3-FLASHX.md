# Plan 175: OpenRouter GLM 5.3 FlashX

## Goal

Add OpenRouter's stable `z-ai/glm-5.3-flashx` route as a hosted model without
changing Geist's provider architecture or allowing the model to reach a local
runner.

## Evidence gate

- Capability value: 4/5
- Evidence quality: 4/5
- Geist fit: 5/5
- Operational safety: 4/5
- Implementation confidence: 5/5
- Total: 22/25

FlashX uses the same GLM 5.3 Flash model family already represented in Geist,
but its dedicated serving tier materially improves latency for interactive and
multi-tool agent loops. The route is new and has only one upstream, so its
performance note and documentation must preserve the launch-date reliability
and routing caveats.

## Implementation

1. Add an explicit hosted `ModelSpec` with the verified OpenRouter model ID,
   context/output limits, modalities, tool support, reasoning requirement, and
   request exclusions.
2. Cover catalog metadata, API discovery, OpenRouter factory routing, local
   load rejection, credential resolution, native tools, and request shaping.
3. Document pricing, capability evidence, operational tradeoffs, and data
   handling without making unverified provider or quality claims.
4. Run focused formatting, linting, typing, backend/frontend tests, API smoke,
   and the Geist pre-push test loop in proportion to this catalog-only change.

## Non-goals

- No new provider, dependency, runner, or database schema.
- No paid inference call.
- No local FlashX loading path.

## Validation

- Ruff lint and format: passed.
- Mypy for the catalog and OnlineAgent: passed.
- Focused catalog, routing, credential, and request tests: 120 passed.
- Affected AgentConfigSection frontend suite: 19 passed.
- Native FastAPI model-discovery smoke: passed.
- Changed-file pre-commit suite: passed.
- The combined model catalog and OnlineAgent run reached 154 passes before 15
  unrelated tests failed because the default SQLite database lacked the
  `chat_session` table.
- Docker smoke: blocked because the local Docker daemon was not running.
- Paid inference: intentionally not run.
