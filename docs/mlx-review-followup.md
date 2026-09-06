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
