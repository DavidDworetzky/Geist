# MLX stack review follow-up

## PR #356: runtime ownership and cache contracts

Review scope: Claude's review of the MLX throughput PR on 2026-09-06.

Accepted fixes:

- Use non-reentrant locks: Starlette can advance and close synchronous generators
  on different worker threads. Serialize runner configuration with the request,
  and explicitly close model iterators when orchestration is interrupted.
- Validate ordinary MLX cache offsets before reuse and retention. Retain no more
  than 32,768 tokens of conversation cache in either decoder. This is a retention
  bound, not an input/output generation limit. Cleanup releases model/drafter/cache
  references. Disabled wrappers after failed automatic setup remain transparent
  until cleanup, avoiding a second mutation of partially installed modules.
- Reject tool markup even with an empty tool catalog, before exposing the marker
  or arguments. Keep ordinary prose streaming with only a marker-sized holdback.
- Do not use DFlash with custom logits processors; name the existing 32-token
  minimum; reject padded recurrent verification and incomplete attention rollback;
  guard zero residual sampling mass. Initialize tuning diagnostics on every path.
- Run CI for the three stacked base branches, and run native adapter contracts on
  macOS plus the Metal suite when the runner reports Metal availability.

Findings not adopted:

- MLX-LM 0.31.3 does not duplicate the final token at the output limit. Its
  generation lookahead consumes each yielded token, including the final token.
  Tests now execute the installed `stream_generate` at limits 1/2/8 and EOS, and
  compare the yielded tokens, text and actual cache offsets.
- The installed sampler filters top-p, then top-k, then temperature. Reordering
  this would change the distribution. Tests now intercept the actual
  `make_sampler` categorical input instead of reproducing its presumed order.
- `ArraysCache.advance` changes padding/length metadata, not a generic offset.
  Both metadata fields must be unset in this supported single-sequence path;
  rollback tests compare metadata as well as recurrent tensors and next logits.
- Exact Qwen 3.8 artifact eligibility is intentional: family aliases or renamed
  checkpoints are not enough evidence to enable a specifically qualified drafter.
  The documented 32-token threshold and pinned dependency pair remain unchanged.

Validation on this Mac, using existing dependencies (no installs):

- Docker, existing `geist-goal-decomposition-backend:latest`, isolated SQLite:
  `pytest tests/agents/test_mlx_llama_runner.py tests/agents/test_dflash_artifact.py
  tests/services/test_chat_orchestrator.py tests/agents/test_chat_template_tools.py
  -q --tb=short -p no:cacheprovider`: **86 passed**.
- Native Apple Silicon, MLX 0.32.2 / MLX-LM 0.31.3:
  `pytest tests/agents/test_mlx_dflash.py tests/agents/test_mlx_llama_runner.py
  -q --tb=short -p no:cacheprovider`: **149 passed**. Tiny real models exercise
  Metal kernels and installed generation contracts; these are not 27B benchmarks.
- Changed-file Ruff passed; mypy on runner and orchestrator passed.
- Full-stack Docker/native browser and real Qwen tool validation is repeated at
  the top of the stack after propagation. The existing user's UI/database is
  preserved while isolated regressions run.

### Real-model follow-up: thread affinity

A stronger native Qwen regression found that fixing lock ownership alone is
insufficient: resuming DFlash on a second worker raised `There is no Stream(gpu,
...) in current thread`. Runner load, completion, stream advancement/close and
cleanup now execute on one persistent model worker. The bridge submits exactly
one `next()` per consumer request; it does not aggregate or eagerly queue output.
Cleanup waits for the active request and then releases the worker. A regression
asserts that different consumer threads still execute all model work/cleanup on
the same thread. Real-model cancellation/recovery and SSE validation are required
again for this follow-up; the earlier green counts do not establish this fix.

Second-review follow-up: load is now serialized, overlapping streams fail busy
instead of self-deadlocking, classifier iterators close explicitly on protocol
errors, cleanup disables retained wrappers, and unverifiable offset-less caches
log a debug diagnostic. Single-slot cache eviction by an opted-in classifier is
an acknowledged performance limitation, not corrupted output; the fourth PR
turns routing off by default. A keyed classifier cache would be a separate
memory/performance policy, not a prerequisite for correct prefix matching.
The retention bound and backpressure behavior are now in the runtime guide.
Decoder-local retention constants stay independently testable. CI stack-base
coverage must stay while these PRs target those bases; removing it before merge
would reopen the verification gap. Existing workflow indentation was normalized
because the repository YAML hook requires it. No dependency versions changed.

## PR #357: measured policy and experimental safeguards

Accepted: independently disable adaptation with `GEIST_MLX_DFLASH_ADAPTIVE=off`
without disabling DFlash/kernels; materialize native hidden/cache state before
calibration timing; bound deferred context with a dedicated 64-item limit; avoid
unused copy-branch allocations and require explicit sampled draft distributions.
Experimental Metal rewrites now require exactly one source match, unsupported
quantization falls back before kernel construction, split-K is validated, empty
expanded-weight reinstall is a no-op even under memory pressure, and conflicting
benchmark flags fail before loading weights. Added CLI help for new experiment
controls and removed the temporary branch name from runtime documentation.

Not adopted: discarding first-round costs, periodic re-probes, or narrowing the
15% deadband would retune a measured policy without new cross-workload evidence.
These remain research options, not established correctness fixes. The cumulative
policy **can** switch from initially fast speculation to later fallback: the
existing 100-slow-round test proves this. Consequently continued round timing is
necessary; removing it after eight rounds would introduce a bug. Single-row
fallback still traverses installed wrappers and is not promised to match the
`GEIST_MLX_DFLASH=off` route's throughput. Archived raw benchmark arrays and exact
historical machine commands are intentionally retained as research provenance.
The opt-in n-gram index is O(prompt + output tokens), not an unbounded cross-run
cache. Review tooling permissions were not widened to bypass denied commands;
Docker/native evidence and the newly enabled CI provide verification instead.

PR #357 integration validation: **88 passed** in the focused Docker runner,
artifact, policy, n-gram and orchestrator suites; **218 passed** in the native
Metal, runner, artifact, policy and n-gram suites after the parent merge.

Second review: native timing materialization is restricted to calibration, so
latched fallback does not synchronize unnecessarily on every token. Explicit
invalid split-K and conflicting benchmark-CLI tests were added. The reviewer's
macOS 14 skip for unsupported experimental `relaxed` Metal math was preserved;
that variant still executes on this newer local Mac. Follow-up native
Metal/policy validation: **162 passed**.

## PR #358: stream failure persistence

Accepted: save already-emitted prose on malformed tool output, disconnect, and
cancel, exactly once with failed/cancelled status and no unvalidated tool calls.
Completed turns remain authoritative and do not duplicate the emitted deltas.
Use the same incremental parser with and without tools, and normalize malformed
closing-marker errors. Probe reset now clears observable state and invalidates
old producers so their cleanup cannot mark a new run closed. Gate waits are ten
seconds, and comments explain worker isolation, constructor bypass and SPA route
ordering. A browser regression checks failed-turn prose survives reload.

Not adopted: swallowing the final parser mismatch would silently reconcile
incompatible output after bytes were sent; it remains an explicit failure.
Bare JSON responses still buffer until EOF for the established whole-response
tool-call compatibility contract. That limitation is documented, not a claim of
universal JSON streaming. The reported 'backend failed to start' copy is not the
orchestrator's mid-stream error path: it already returns 'Chat completion failed'.
Small explicit iterator cleanup and private-buffer white-box assertions remain
intentional; neither needs a new abstraction or public parser API.

PR #358 focused Docker validation: **151 passed**. Isolated Docker startup was
clean; Chrome **5/5 chat tests passed**, including gated streaming and failed-turn
prose persistence across reload. Its frontend was the existing top-of-stack
build (no frontend production code changed in this review pass).

## PR #359: XML protocol hardening

Unwrapped `<function=...>` now fails closed in both complete and one-character
streaming paths, instead of exposing tool arguments as prose. XML requires a
matching offered schema; missing schema fails loudly instead of silently turning
numeric-looking strings into integers. The optional argument remains for JSON
compatibility, where schemas are not needed to preserve JSON types. Boolean
`additionalProperties` is normalized before type inspection. Nullable strings
retain whitespace around `null`; exact `null` still represents None, because
Qwen renders those two values identically and choosing string would break actual
null round trips. Multiple functions inside one wrapper fail with a clearer
diagnostic. Regression tests cover these boundaries and actual-model cross-worker
stream close followed by another generation.

Intent routing remains off for unset existing workspaces as explicitly requested;
saved true remains an opt-in, even though it is uncommon in historical settings.
Process-scoped test search stubs and test assertions are not production code.
The first actual-model cross-worker test exposed the deeper Metal affinity bug;
that fix belongs in #356 and is propagated here before final qualification.

## Integrated qualification after all parent merges

Final production-code head before this evidence-only update: `71c7345`.
All tests used existing dependencies; no packages were installed or updated.

- Docker backend: **592 passed, 7 skipped** with isolated SQLite, running
  `pytest tests/agents tests/services/test_chat_orchestrator.py
  tests/services/test_tool_intent_router.py tests/test_streaming_probe.py
  tests/services/test_user_settings_service.py tests/api/test_user_settings_routes.py
  tests/services/test_tool_registry.py -q --tb=short -p no:cacheprovider`.
  Docker does not substitute for the skipped native/live-model paths below.
- Native Apple Silicon: **228 passed**, running `pytest
  tests/agents/test_mlx_dflash.py tests/agents/test_mlx_llama_runner.py
  tests/agents/test_dflash_artifact.py tests/agents/test_speculation_policy.py
  tests/agents/test_ngram_draft.py -q --tb=short -p no:cacheprovider`.
- Actual installed Qwen 3.8 27B 4-bit with DFlash2: **3 passed** in 19.69 seconds,
  using `GEIST_RUN_MLX_TOOL_SMOKE=1 pytest tests/agents/test_mlx_tool_live.py -q -s
  --tb=short -p no:cacheprovider`. This verifies the tokenizer's tool format,
  generated tool dispatch/result use, and cross-consumer-worker stream
  advancement/close followed by another generation.
- Frontend: **42 passed** across `useCompleteText`, `src/__tests__/Chat`,
  `Settings`, and `useUserSettings`, with the existing React Scripts runner and
  `--watchAll=false --runInBand --runTestsByPath`. Existing React `act` warnings
  and intentional error-path console messages remain test-only warnings.
- Isolated Docker application at port 5592: successful startup and authenticated
  `curl /chat` HTTP 200. Chrome `playwright test` against all three E2E files:
  **11 passed**, including source-gated streaming, Qwen XML dispatch,
  failed-prose reload, memory privacy, and router default/opt-in/opt-out. Repeating
  memory tests against a reused database caused duplicate-fixture failures;
  recreating the disposable test container restored a clean full pass without
  changing production code or weakening assertions.
- Real native application at port 5593: production `create_app`, existing model
  artifact, offline weights, router off, fresh SQLite and no model stubs. Chrome
  verified generated prose while Stop remained visible, cancellation, reopening
  the persisted partial response, a successful follow-up generation, and saved
  router opt-in/opt-out. No browser errors. Cancellation aborts the SSE connection
  before its final navigation event, so this check explicitly reopened the saved
  chat from persisted history. A single observed first-visible latency was 10.45
  seconds; this is a smoke result, not a throughput/latency benchmark.

Runtime isolation: Docker used the existing backend image and an explicit
scratch Compose file. Native used a scratch Python launcher with `MLX_BACKEND=1`
and dotenv loading disabled; `make run` was not invoked because this pass avoided
implicit setup/install steps and local secret-file reads. Existing frontend
assets were reused because this review pass did not change frontend production
code. Ports 3000/5587 and the user's existing database were not replaced.
Missing optional SendGrid/Twilio credentials produced adapter warnings, not chat
or model failures. All review-fix commits passed the applicable local hooks.
