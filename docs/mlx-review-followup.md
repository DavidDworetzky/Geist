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

Third-review follow-up: worker-side iterator finalization now executes inline
instead of waiting on its own single-thread queue. Worker creation has its own
lock, failed initial loads release the executor, and a public-load regression
compares constructor, decode, close and cleanup thread identities. Load and
non-streaming completions share the explicit busy policy; streamed contention
fails on first advance. Cleanup remains a safe wait, but stale-agent phase-out
and new-model construction no longer hold the shared agent-cache lock. A
separate nonblocking local-model creation lock prevents duplicate loads while
allowing online lookups; a stalled-cleanup regression pins both behaviors.
Focused Docker: **90 passed**; native Metal/runner: **156 passed**.

The reported process-global generation-stream collision did not reproduce.
`new_thread_local_stream` returns a `ThreadLocalStream` dispatcher, not a concrete
stream tied to its construction thread (see the [MLX API reference](https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html)).
A new real-Metal regression loads two actual adapters through public `load()` on
distinct workers, suspends runner A, loads/advances/closes/cleans up B, and resumes
A to completion with live lookahead/cache state. It passes without changing the
backend's stream configuration. The original affinity failure involved moving
live lazy-array/DFlash state between consumer threads, which worker pinning fixes.
The sentinel-based generator advance is safe under PEP 479; changing it would
not fix an established defect. Cosmetic dependency spacing remains untouched.

Hook exception for this follow-up: the isolated mypy hook reports the unchanged
`adapters/whisper_adapter.py:38` returning Any, and Bandit reports the unchanged
`app/main.py` script entrypoint's `0.0.0.0` bind. Both reproduce on the committed
`0b5438f` version of `app/main.py` with the same cached hook executables. The
changed runner and app pass mypy with the installed project dependencies; Bandit
reports no new finding. Those two hooks were skipped only for this commit after
baseline comparison; all other applicable hooks ran, and no hook configuration,
adapter, dependency, or bind address was changed to silence them.

Fourth-review follow-up (T): a duplicate readiness request must not turn an
in-flight local model load into a failed UI state. A typed busy result now carries
the model being loaded. Readiness requests for that same model follow the owner's
status without overwriting a terminal result; conflicting model requests still
fail explicitly because they are not queued. Publishing the completed agent also
publishes ready, closing the race where a repeated start resets loading after
the constructor reports ready but before a chat-owned load enters the cache.
The creation lock is released even if stale-cache clearing fails.

Concurrent regressions cover both chat-owned and readiness-owned loads, including
the constructor-ready/cache-publication window, one actual construction, preserved
ready/failed outcomes, and a conflicting target without an owner. Docker:
`pytest tests/agents/test_local_agent_cache.py tests/agents/test_model_load_status.py
tests/services/test_local_models.py tests/agents/test_mlx_llama_runner.py -q
--tb=short -p no:cacheprovider`: **104 passed**. Ruff and installed-project mypy
on `app/main.py` and the models endpoint pass. Bandit still reports only the
unchanged script bind above; the same scoped hook exceptions apply to this fix.

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

Third-review follow-up: explicit profiling now fences native hidden/cache state
even after calibration; default unprofiled fallback still avoids that extra
fence. A real-Metal regression asserts zero such fences in ordinary fallback,
one per token in profiling, and no re-calibration. Added help for the three
documented hands-on flags (`--small-m`, `--autotune`, `--split-k`). Native
Metal/policy follow-up: **165 passed**. Unsupported macOS 14 `relaxed` experiments
continue to report Metal's capability error rather than silently changing the
requested math variant; this is an explicit research mode, not the default path.

PR #357 integration validation: **88 passed** in the focused Docker runner,
artifact, policy, n-gram and orchestrator suites; **218 passed** in the native
Metal, runner, artifact, policy and n-gram suites after the parent merge.

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

Second-review follow-up: historical tool names are serialized but no longer
included in the output parser's offered-name map. This covers both empty and
narrowed tool catalogs across adapters, not just MLX. Orchestration independently
rejects any completed call that was not offered before persisting or dispatching
it. Tests cover history-bearing no-tools/narrowed-tool requests and a malicious
structured backend under disabled tools and privacy-sensitive routing.
Model-message snapshots now use the same lock as cancellation writes; probe
start/state are locked, and the malformed-output fixture fails explicitly if the
expected parser error disappears.

The persistence lock still spans the final database write intentionally: moving
that write outside requires an in-flight persistence state and retry semantics
to preserve exactly-once behavior. The write occurs when ending/cancelling the
turn, not during ordinary steady-state token production. Incremental early
rejection and whole-response incomplete-markup diagnostics can retain different
wording without relaxing either fail-closed contract. Shared prefix-holdback
logic for different stop/protocol markers is not refactored here.
