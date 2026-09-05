# MLX DFlash runtime

Geist can draft and verify Qwen 3.8 27B tokens in-process using the existing
MLX model, a DFlash 2 auxiliary checkpoint, and small-row Metal matmul kernels.
There is no external inference server or backend migration. The target remains
the original 4-bit model; only the auxiliary drafter is quantized to 8-bit.

## Installation and selection

Install the optional native MLX dependencies through the project's approved
dependency workflow, then explicitly download the auxiliary weights:

```sh
.venv/bin/python scripts/download_mlx_dflash.py
```

This fetches approximately 3.85 GB from `incoai/Qwen3.8-27B-DFlash2`, pinned to
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Its path is underneath Geist's existing
model home, in `auxiliary/qwen3.8-27b-dflash2/<revision>/snapshot`.
The runtime never downloads the drafter during generation.

`GEIST_MLX_DFLASH=auto` is the default. It enables the optimization for
`Qwen/Qwen3.8-27B` when the auxiliary snapshot exists, and otherwise keeps the
ordinary MLX-LM path. Requests with an output budget below 32 tokens use ordinary
decoding. Invalid/incompatible artifacts or a failed kernel qualification fall
back with a warning in automatic mode.

- `GEIST_MLX_DFLASH=off`: force ordinary MLX-LM decoding.
- `GEIST_MLX_DFLASH=on`: require DFlash; surface initialization errors.
- `GEIST_MLX_DFLASH_DIR=/absolute/path`: select an explicit local snapshot.
- `GEIST_MLX_PREFILL_STEP_SIZE=2048`: configure prompt-processing chunk size.

Restart the loaded runner after installing an auxiliary artifact or changing
these settings. The first DFlash request includes checkpoint loading, 8-bit
conversion, kernel compilation, and shape qualification. Warm decode timing
does **not** include that one-time cost.

## How the speedup works

1. Capture target hidden states at layers 5, 19, 33, 47, and 61.
2. Predict seven candidate tokens in a parallel eight-position draft block.
3. Verify the anchor and candidates together, sharing target-weight reads.
4. Accept the matching prefix and produce a corrected or bonus token.
5. Roll back full-attention KV and replay only the rejected block's retained
   Gated DeltaNet recurrence inputs—not the entire target forward pass.

The Metal kernel dequantizes 64-value weight groups into threadgroup memory,
reuses them across up to eight token rows with `simdgroup_matrix` operations,
accumulates in float32, and reduces split-K partial sums. Per-model wrappers
avoid global monkeypatching. Single-token operations retain the native kernel.
At load time, real projection shapes qualify against native QMM and choose
between split-K layouts; shapes without at least an 8% measured advantage or
within-tolerance numerical agreement keep native QMM.

Draft and target caches survive an exact conversation-prefix match. Branching,
cancellation, and mismatching tokenization invalidate reuse. Output-limit and
EOS truncation retain the correct recurrent state, including when either cuts
through an accepted draft block.

## Correctness and limitations

Greedy decoding accepts only a prefix matching the target verifier's argmax.
Sampled decoding uses probability-ratio acceptance and the positive residual
target distribution on rejection, including tokens absent from the draft's
candidate set. Filtering follows MLX-LM's order: top-p, then top-k, then
temperature. The Qwen runner retains its top-k=20 policy on both paths.
Stop strings are buffered across text chunks, and tool-enabled turns retain
the existing structured response parser.

The algorithm preserves the target distribution in exact arithmetic. It is
**not a promise of bit-identical output** to single-token MLX decoding: batched
matmuls, dequantization, and split-K reduction change floating-point rounding.
Greedy tokens can differ at close decisions. The benchmark records exact-token
agreement and the first divergence instead of hiding this.

The current implementation is single-sequence, dense Qwen 3.8, unsharded, with
its matched DFlash 2 checkpoint. MoE, other model identities, batching, and
arbitrary MTP heads are not enabled. Low draft acceptance can eliminate the
speedup or cause a regression; larger output limits alone do not raise token/s.
Automatic selection is based on artifact availability and kernel qualification,
not a claim that every prompt is faster. Use `off` for strict control runs.

## Reproduce the paired benchmark

```sh
.venv/bin/python scripts/benchmark_mlx_dflash.py \
  --weights-dir /absolute/path/to/qwen3.8-27b-4bit/snapshot \
  --drafter-dir /absolute/path/to/dflash2/snapshot \
  --small-m --autotune --drafter-bits 8 \
  --max-tokens 256 --trials 3 --output /tmp/mlx-dflash.json
```

The same loaded target and tokenizer run both paths. Each path warms up;
baseline wrappers are disabled. Every timed request has a fresh conversation
prefix. Throughput is `(emitted_tokens - 1) / time_after_first_token`, excluding
prefill. Reports also include complete elapsed time, acceptance, memory, and
token hashes. These are single-stream decode numbers, not prefill token/s or
aggregate batching throughput. Other workloads on the Mac can affect timings.

For stochastic testing add `--temperature 0.7 --top-p 0.9 --top-k 20`; token equality is not
expected between independently sampled runs. `scripts/benchmark_mlx_small_m.py`
isolates projection shapes. Half operands and wider column tiles are benchmark
experiments only; automatic runtime tuning does not enable them.

Native correctness tests are in `tests/agents/test_mlx_dflash.py`; routing,
fallback, and consuming streaming-detokenizer behavior are checked separately
in `tests/agents/test_dflash_artifact.py`.

## M2 Max 96 GB results — September 4, 2026

MLX 0.32.2, MLX-LM 0.31.3, 256 generated tokens, three warm paired trials per
prompt, greedy decoding, target revision `3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.
Reported rates are medians; gains are medians of paired speedups.

| Prompt | Ordinary MLX tok/s | DFlash + Metal tok/s | Gain |
| --- | ---: | ---: | ---: |
| Python LRU implementation | 21.13 | 33.67 | 59% |
| Quadratic-formula derivation | 21.05 | 37.68 | 79% |
| Sky/scattering explanation | 20.98 | 22.34 | 7% |

Peak observed decode rate was 37.75 tok/s. All three 256-token math outputs
matched the baseline exactly. Code first diverged at token index 215 and prose
at index 89, illustrating the floating-point caveat above. Observed process
peak MLX allocation, including drafter conversion, was 20.82 GB.

Earlier experiments: plain DFlash with native small-M QMM achieved only
19.16 tok/s versus a 20.86 tok/s control on a 128-token code sample. The first
Metal verifier raised that to 29.21 tok/s with identical tokens. Wider column
tiles and half operands did not demonstrate an end-to-end advantage and remain
outside automatic tuning. Full-BF16 drafting did not beat the chosen 8-bit
configuration on this suite. Two- and four-token draft caps were generally
slower than seven.

Raw repeated results: `benchmarks/mlx/2026-09-04-m2-max-greedy.json`. The actual
Geist runner also passed a two-turn check: it streamed text, remembered
"cobalt", and reused 171 of 201 prompt tokens on the follow-up. Cold-start
latency is separate and can substantially exceed warm benchmark prefill time.

These results are not measurements on a Mac Studio, and do not establish a
170 tok/s ceiling or achievement. Multi-token verification amortizes weight
reads; attainable speed still depends on acceptance, compute throughput,
memory bandwidth, context length, and launch/recurrent-state overhead.

One additional paired trial at temperature 0.7/top-p 0.9 measured:

| Prompt | Ordinary MLX tok/s | DFlash + Metal tok/s | Gain |
| --- | ---: | ---: | ---: |
| Python LRU implementation | 21.13 | 33.92 | 60% |
| Quadratic-formula derivation | 21.14 | 35.56 | 68% |
| Sky/scattering explanation | 21.02 | 20.56 | -2% |

This is a stochastic spot-check, not a repeated statistical estimate. It shows
both the useful code/math speedup and the low-acceptance prose limitation.
See `benchmarks/mlx/2026-09-04-m2-max-sampled.json`.

The historical stochastic report used top-k=0. Use `--top-k 20` to reproduce
the current Qwen runner's sampling policy; that historical report is not a
measurement of the new top-k=20 path.

### PR integration qualification

After integrating with main at `8afc32d`, a fresh one-trial, 256-token greedy
check measured code **21.06 → 34.31 tok/s** and math **20.92 → 33.90 tok/s**.
See `benchmarks/mlx/2026-09-04-m2-max-pr-main.json`. These are spot checks, not
new repeated medians. Kernel qualification selected a different split-K mix;
code diverged at token index 215 and math at index 24. The earlier exact math
matches therefore do not establish universal bit-identical behavior.

The 131,072-token catalog ceiling is configuration, not a validated full-length
generation result. Neither this check nor the earlier 1,024-token tests prove
performance or stability at that ceiling.

The integrated runner also completed a sampled two-turn check at temperature
0.7/top-p 0.9/top-k 20. It returned the remembered word "cobalt" and reused 105
of 132 prompt tokens. That short semantic/cache check is not a speed benchmark.

#### Pre-push checks

Validation used the already installed development environment and existing
Docker image; no packages were installed during PR preparation.

- `ruff check .`: passed.
- `ruff format --check` on all 16 changed Python files: passed.
- `mypy --config-file=pyproject.toml .`: passed, 181 source files. The existing
  repository override excludes adapted `agents.architectures.llama.*` code
  from type errors; native tests qualify that code separately.
- `python scripts/verify_dependency_policy.py` and `uv lock --check --offline`:
  passed.
- `pre-commit run` on the staged PR: passed, including supply-chain policy,
  Ruff lint/format, mypy, Bandit, and staged secret scanning. All hook
  environments were already cached.
- `pytest tests/agents/test_mlx_dflash.py -q`: 98 passed on native Metal.
- The following focused suite passed both natively and in Docker: 219 tests.

```sh
pytest tests/agents/test_dflash_artifact.py \
  tests/agents/test_mlx_llama_runner.py tests/agents/test_model_catalog.py \
  tests/agents/test_new_architecture.py tests/agents/test_generic_completion.py \
  tests/agents/test_native_tool_turns.py tests/services/test_local_models.py \
  tests/services/test_user_settings_artifacts.py \
  tests/services/test_user_settings_service.py -q
```

The Docker run used `geist-goal-decomposition-backend:latest` (existing Linux
amd64 image), a read-only PR checkout at `/work`, `PYTHONPATH=/work`, and
`python -m pytest` with the same arguments plus `-p no:cacheprovider`. This
checks Linux contracts with that image's installed dependencies, not a fresh
image build or Metal execution.

An isolated backend startup with the PR checkout, a disposable SQLite database,
and writable temporary application-data directories completed successfully.
`curl -fsS -o /dev/null -w 'docs HTTP %{http_code}\n' http://127.0.0.1:5586/docs`
and the corresponding `/openapi.json` request both returned HTTP 200. Initial
attempts used a nonexistent image virtualenv path and then a read-only data
directory; correcting the test setup resolved both, without runtime changes.

The standard `docker compose up -d --build`, port-3000 browser chat/settings
loop, and `make run MLX_BACKEND=1` were **not completed** for this PR. Other
worktrees own the standard ports; the clean PR worktree has no installed
environment/frontend build, and package installation was not authorized.
The existing services and local `.env` files were left untouched. Direct
native model generation and isolated Docker API startup are the evidence here,
not a claim that the full UI/native-app loop passed.

Dependency audit used the locked all-extras export and existing pip-audit 2.10.1:

```sh
uv export --locked --offline --all-extras --no-hashes --no-emit-project \
  --no-header --no-annotate --output-file /tmp/geist-mlx-pr-audit-requirements.txt
pip-audit --requirement /tmp/geist-mlx-pr-audit-requirements.txt --no-deps \
  --disable-pip --format json --output /tmp/geist-mlx-pr-audit.json
```

The audit exited nonzero: 80 advisory records across 10 unchanged packages
(`aiohttp`, `datasets`, `pytest`, `python-dotenv`, `python-multipart`, `requests`,
`sentencepiece`, `setuptools`, `starlette`, `torch`). The only changed package
versions are MLX and MLX-Metal, both 0.32.2; neither had reported advisories,
nor did MLX-LM 0.31.3. This is not a clean full-dependency audit; unrelated
dependency remediation is outside this optimization PR.

## First-principles follow-up

The next pass did **not** establish a new peak above 37.75 tok/s. It tested
several distinct ways to reduce the cost per accepted token rather than merely
raising the output limit. All kernel-layout and selector experiments remain
opt-in; the automatic runtime still uses the trained eight-position block and
the original staged kernel with an 8-bit drafter.

The useful objective is:

`emitted tokens per second = E[accepted draft tokens + correction/bonus] / (draft + verify + rollback time)`

Fenced profiling of two 128-token prompts attributed about 86–87% of measured
round time to target verification, about 12% to drafting, and around 1% to
rollback. Those profiling fences add overhead, so use the proportions as a
bottleneck diagnosis, not the profiled throughput as a headline benchmark.
Eliminating rollback entirely cannot produce a large speedup.

### Experiments and what they ruled out

| Experiment | Observation | Decision |
| --- | --- | --- |
| Expand real Q4 projections to BF16 in memory | Extra bandwidth generally outweighed removing unpacking; head ~17.5 ms versus ~6.9 ms staged Q4 | Do not expand the target |
| Tile-major compressed packing | No convincing screening advantage | No extra production weight copy |
| Direct MMA register fragments, with and without SIMD shuffles | Correctness tests matched the staged kernel, but head ~11–13 ms | Keep shared-memory staging |
| Stage 128 K values per barrier; pad shared-memory rows | Larger staging slowed the head; padding did not establish a win | Keep 64-value, unpadded staging |
| Exact 16-value dequantization lookup tables | Preserved rounded weight values, but additional table traffic did not win | Do not allocate production lookup tables |
| Wider 16/32-position speculative blocks | Code accepted ~6.1/~6.5 drafts per round versus ~5.1 at width 8, while throughput fell from 33.66 to 22.65/14.27 tok/s | Do not extrapolate an eight-position checkpoint to longer blocks |
| Expected-prefix and maximum-sequence-probability selector paths | Code saved one round, math saved none, prose needed more rounds | Keep local greedy selection |
| Quantize only the auxiliary drafter to 4-bit | Code/math needed one fewer round, prose two more; variable-load rates were 31.23/34.87/20.66 tok/s | No established broad win; keep 8-bit default |

Projection measurements above were preliminary screenings of independent calls
over up to three rotated real matrices. They can hide serial dependencies and
are not sufficient qualification for production. The shape tuner and projection
benchmark now time dependent chains instead: every projection consumes data
from its predecessor. Runtime qualification uses 12 calls per timed chain,
separate numerical checks, and the existing 8% advantage threshold. This fixes
the measurement method; it is not itself a claim of faster decoding.

The exploratory harness supports packed layouts, direct/shuffled fragments,
staging sizes/padding, BF16 expansion, lookup tables, 1–32-row tiles, non-power-of-two
split-K divisors, wider draft blocks, and selector lookahead. None is selected
automatically. `--independent` explicitly restores the old projection-screening
method; dependent timing is now the benchmark default. Exhaustive small-lattice
tests check both lookahead objectives, and native tests cover their verified
generation, profiling, and cache behavior.

A final real-weight dependent-chain smoke check reproduced roughly 1.8–1.9x
projection speedups for the existing staged kernel, with all tested relative
maximum errors below 0.006. See
`benchmarks/mlx/2026-09-04-m2-max-serial-projections.json`. This checks the revised
measurement harness; it does not promote any of the exploratory layouts.

Raw exploratory evidence is in
`benchmarks/mlx/2026-09-04-m2-max-experiments.json`. A repeated 256-token control
after serial qualification measured medians of 32.97 code, 35.73 math, and
21.97 prose tok/s; the best math trial was 37.41 tok/s. Other work on this Mac
was not suspended: a busy Python process from another Geist worktree was
observed, and some earlier native controls varied from 16 to 21 tok/s. Small
timing differences are inconclusive. See
`benchmarks/mlx/2026-09-04-m2-max-serial-qualification.json`.

The auxiliary 4-bit spot-check is recorded separately in
`benchmarks/mlx/2026-09-04-m2-max-q4-drafter.json`. It reduced process peak MLX
allocation during this run from approximately 20.82 to 19.96 GB, including
conversion, but needs controlled repeated comparisons before runtime adoption.

A separate 1,024-token paired trial with the final 12-call tuner sustained:

| Prompt | Ordinary MLX | DFlash + Metal | Agreement |
| --- | ---: | ---: | --- |
| Code | 20.76 tok/s | 35.25 tok/s | First difference at index 215 |
| Math | 20.71 tok/s | 37.38 tok/s | All 1,024 tokens identical |

This validates longer sustained decoding, not a new same-workload peak or a
131,072-token run. See `benchmarks/mlx/2026-09-04-m2-max-1024.json`.

### Physical bounds versus the current implementation

Inspecting the pinned safetensors headers separates 14,417,640,448 bytes of
non-embedding language-model tensors, 715,161,600 embedding bytes, and
921,460,192 vision bytes. The earlier 16.05 GB snapshot-size estimate counted
vision and the complete input embedding table, which text decode does not read
in full on every pass. About **14.42 GB** is a better first-order estimate for
the target's streamed text-decoder weights, including its separate output head.

Apple specifies [400 GB/s for M2 Max](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/).
The [2025 Studio specifications](https://support.apple.com/en-lamr/122211)
list 546 GB/s for the top M4 Max and 819 GB/s for M3 Ultra. The 512 GB RAM
configuration refers to **M3 Ultra**, not M4 Max; this is a hardware comparison,
not a claim about current purchase availability.

| Chip | Ideal target-weight read | Weight-only ordinary decode | Perfect 8-token target-only bound |
| --- | ---: | ---: | ---: |
| M2 Max | 36.0 ms | 27.7 tok/s | 222 tok/s |
| M4 Max, 40-core GPU | 26.4 ms | 37.9 tok/s | 303 tok/s |
| M3 Ultra | 17.6 ms | 56.8 tok/s | 454 tok/s |

These are deliberately optimistic bandwidth calculations, **not attainable
speed predictions or universal physics ceilings**. They assume one weight read
per block at advertised peak bandwidth, perfect acceptance in the last column,
and zero draft, compute, KV/state, activation, synchronization, or contention
cost. Larger blocks change this bound again and eventually encounter a compute
limit. More RAM alone does not increase memory bandwidth.

The successful 256-token math run emits about 6.7 tokens per roughly 178 ms
round. Perfect acceptance at the same round cost reaches only about 45 tok/s.
Consequently, 100 tok/s needs a round around 67 ms and 170 tok/s needs about
39 ms at the same acceptance. On M2 Max, the latter barely exceeds the ideal
target-weight read time before paying for the drafter or arithmetic. It is an
extremely aggressive optimization target, not a number this runtime has reached.

### Where a genuinely larger win could come from

1. **Branch-aware verification within the same matrix tile.** Spend rows on
   alternate continuations at uncertain tokens instead of on a suffix discarded
   after the first mismatch. The selector already supplies a transition lattice.
   Qwen's hybrid architecture requires per-branch DeltaNet state and convolution
   history, depth-based RoPE, ancestor-only attention masks, and exact cache
   compaction of the accepted path. A plain causal mask is incorrect. This is
   a proposed backend experiment, not an implemented feature or guaranteed gain.
2. **A drafter trained for longer useful blocks.** The measured wider-block
   failure is an acceptance problem as well as a GEMM problem. Train/evaluate
   against accepted-prefix length per measured verifier millisecond, not merely
   token-level prediction accuracy. More speculative rows without more correct
   tokens increase work.
3. **A substantially better small-row quantized pipeline.** Double-buffered
   weight loading/dequantization and a measured tile-occupancy model remain
   plausible experiments. The failed register/layout variants show why fewer
   source-level instructions or barriers do not by themselves imply less GPU
   time. GPU counter evidence is needed before another speculative rewrite.
4. **Continuous batching for aggregate throughput.** Independent requests can
   share a weight read without prediction errors, but aggregate tokens/s is a
   different metric from this single-user stream. It cannot be presented as a
   faster individual conversation.

## Validation status

The local-model and Geist test-loop skills guided pinned-artifact placement,
native inference checks, and separate reporting of application smoke coverage.

Passed (149 tests):

```sh
.venv/bin/pytest tests/agents/test_dflash_artifact.py \
  tests/agents/test_mlx_llama_runner.py tests/agents/test_model_catalog.py \
  tests/agents/test_new_architecture.py tests/agents/test_generic_completion.py \
  tests/agents/test_native_tool_turns.py tests/services/test_local_models.py \
  tests/services/test_user_settings_artifacts.py \
  tests/services/test_user_settings_service.py -q
```

Passed on native Metal (102 tests after the first-principles follow-up):

```sh
.venv/bin/pytest tests/agents/test_mlx_dflash.py tests/agents/test_llama_mlx.py -q
```

Targeted `ruff check`, `ruff format --check`, `git diff --check`, and
`uv lock --check --offline` also passed. The earlier dependency audit reported
pre-existing advisories in unrelated dependencies; this work did not remediate
them. No dependency advisory was reported for the MLX update.

Full-application smoke remains **unverified**. A previous Docker build attempt
hit the existing Linux-aarch64/lockfile environment mismatch; this worktree has
no running Compose services. Port 5001 is occupied by another worktree's Docker
backend. Starting an isolated native server on port 5017 was then blocked by
the permission reviewer because `app.main` invokes `load_dotenv()`. No local
`.env` was inspected or modified, and startup-only permission was requested.
`make run MLX_BACKEND=1` and browser chat/settings against this changed backend
therefore remain outstanding. The existing frontend on port 3000 returned HTTP
200, but belongs to another worktree and is not counted as validation here.

The changed code and evidence are local and uncommitted; existing services and
unrelated skill-directory changes were left untouched.
