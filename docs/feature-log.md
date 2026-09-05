# Feature research log

## 2026-09-05 — MLX single-stream inference throughput

### Question and experimental contract

Can Geist's Qwen3.8-27B runtime approach 100 output tokens/s on an M2 Max 96 GB
without switching away from MLX? Which improvements also help ordinary code,
math and prose, rather than just a favorable copying workload?

Start: PR #356, `eef0ae3`. Target: the pinned Q4 MLX checkpoint; drafter: the
matched DFlash 2 checkpoint quantized to eight bits. Environment: MLX 0.32.2,
MLX-LM 0.31.3. No dependencies, model downloads, or global machine settings
changed during this pass. The original worktree and published PR were kept
separate from the experimental branch `codex/mlx-100tps-lab`.

Decode throughput is `(output tokens - 1) / seconds after the first token`.
It excludes prefill and cold model/compilation setup. Request elapsed time,
token agreement, acceptance, memory and prefix-cache hits are recorded
separately. Batch throughput and prompt-processing tokens/s are not substituted
for single-stream output throughput.

### Research sequence and decisions

| Experiment / hypothesis | Result | Decision |
| --- | --- | --- |
| In-process DFlash shares target-weight reads across eight positions | Native small-row QMM alone could erase the benefit | Pair drafting with qualified small-row kernels |
| Staged Q4/Q8 Metal matmul, split-K chosen per real matrix shape | Earlier code/math runs rose from ~21 to ~34–38 tokens/s | Retain as the general-purpose foundation; acceptance still matters |
| Time dependent projection chains rather than independent launches | Exposed misleading overlap/cache effects in early screens | Keep the corrected qualification methodology |
| Pad rows to unlock different dispatch; relaxed/fast math | No substantial improvement over the staged kernel | No new default |
| Half accumulation or moving affine scale/bias outside the dot product | Precision changes; mixed or negative shape results | Keep as experiments only |
| Compile recurrent layers with explicit rollback state | Small eight-row microbenchmark win, but effectively tied in the broader repeated suite | Do not select compilation as a general speedup |
| Propose long spans already present in the conversation, then verify them | 64–96 tokens/s in initial copy-edit screens | Useful specialized route, not a general chat result |
| Count the anchor when choosing a matmul shape | 128 rows ~1.01 s versus 129 rows ~1.31 s; 127 proposals avoid the cliff | Use aligned proposal limits in the specialized benchmark |
| Spend spare RAM on BF16-expanded MLP weights for wide blocks | 128-row verifier ~1.01 to ~0.87 s; ~34.2 GB extra weights | Opt-in, bounded-memory wide-block route only |
| Preserve source position across successfully copied spans | Complete-file copying rose from 78.43 to 120.50 median tokens/s | Retain source continuity with mandatory target verification |
| Expand prose beyond one explanatory prompt | Essays/stories regressed to ~11–12 tokens/s with unconditional DFlash | Add and qualify a per-request break-even fallback |

### General baseline before adaptive fallback

Nine prompts: LRU implementation, JSONL validation/ranking, worker pool;
quadratic derivation, Bayes calculation, induction proof; scientific
explanation, original essay, original story. Two trials per prompt, 256 output
tokens, greedy decoding, fresh prefixes, no copy proposals or expanded weights.
Verification widths 1–8 were warmed for both implementations before timing.
Each task-type median below contains six measurements.

| Task type | Ordinary MLX tokens/s | DFlash + qualified Metal tokens/s | Plus compiled recurrence tokens/s |
| --- | ---: | ---: | ---: |
| Code | 19.91 | 27.69 | 27.57 |
| Math | 19.01 | 29.30 | 29.33 |
| Prose | 18.97 | 11.77 | 11.87 |

This is why the fastest copy benchmark cannot be advertised as general
throughput. Compilation changed the aggregate result by less than 1%, with
mixed per-prompt results. It did not solve low draft acceptance. The speculative
variants matched one another's token hashes for each prompt; equivalence to
single-token greedy decoding was not universal because reduction order can
change close decisions. This suite measures speed, not broad task accuracy.

Raw evidence:
[general baseline and compilation](../benchmarks/mlx/2026-09-05-lab/general-extended-before-adaptive.json).

### Specialized result: complete-file editing

The prompt contains a repetitive Python configuration file and requests one
constant change while preserving everything else. With verified copying,
source continuity, 127 proposals plus one anchor, and wide BF16 MLP weights:

- Three runs: **120.50, 108.72, 126.53 tokens/s**; median **120.50**.
- Each complete response has 1,802 tokens including EOS, matching its ordinary
  decoder control exactly. The extracted Python parses and matches the exact
  requested one-line edit, including whitespace.
- Median ordinary decoding: **18.40 tokens/s**. Median request elapsed time:
  **112.29 to 30.85 seconds**, including prefill but excluding cold setup.
- Peak observed MLX allocation: **55.29 GB**. No prefix-cache hits were used.
- This is one synthetic copying workload repeated three times, not a claim of
  100 tokens/s on novel prose, mathematical reasoning, or code synthesis.

Raw evidence:
[complete-file repeated result](../benchmarks/mlx/2026-09-05-lab/copy-continuation-repeated.json),
[prompt](../benchmarks/mlx/2026-09-05-lab/copy-edit-prompt.txt), and
[earlier full-file result before source continuity](../benchmarks/mlx/2026-09-05-lab/copy-complete-before-continuity.json).

### Adaptive hypothesis

Emit four real tokens with ordinary MLX to measure per-request single-token
cost, then compare cumulative speculative performance against that reference.
The first version used four rolling rounds and a 5% required advantage. It
abandoned useful speculation on code/math after transient difficult spans and
is rejected. The revised gate requires eight rounds and at least a 15%
cumulative slowdown before switching for the rest of the request. No guessed
task label or alternate model is used. Calibration tokens and failed probes
count toward elapsed time and output limits.

The target cache is preserved. Deferred drafter context is bounded and retained
so a follow-up can still reuse its prefix safely. Native tests cover transitions
at output limits, greedy and sampled generation, rotating draft caches, EOS,
cancellation, and follow-up reuse. This gate is independently selectable;
copying, expanded weights and compilation are not required by it.

The revised gate's first three-prompt screen kept code and Bayes drafting
enabled, at a 3–4% cost, and raised creative-story throughput from 10.97 to
15.37 tokens/s. Ordinary MLX still achieved 17.09 on that story. This is a
mixed-workload tradeoff, not a speed guarantee. The historical rolling gate
and the revised screen are retained separately:
[rejected rolling gate](../benchmarks/mlx/2026-09-05-lab/general-adaptive-rolling-v1.json),
[cumulative gate screen](../benchmarks/mlx/2026-09-05-lab/general-adaptive-cumulative-v2-screen.json).

### Selected mixed-workload baseline: adaptive DFlash + qualified Metal

Select the existing shape-qualified small-row Metal kernels plus the
`cumulative-eight-v2` fallback as the **experimental mixed-workload baseline**.
Use the Q4 target, eight-bit auxiliary, eight-position block, seven proposals,
and local selector. Copying, expanded weights, and compiled recurrence are off.
At the end of the research pass this was an opt-in candidate. The subsequent
user-requested branch-default promotion is recorded below; the application now
enables adaptation automatically. The ablation harness still uses explicit
`--small-m --autotune --adaptive` flags for reproducible comparisons.

The kernel work is the broadly reusable foundation. The new fallback addresses
the largest regression found in this pass without requiring a task classifier
or another backend. It is not a universal improvement over ordinary MLX or
unconditional DFlash: calibration and synchronized measurements cost time, and
the first four tokens also change subsequent speculative block alignment.

#### Final repeated task-type benchmark

The same nine prompts, two trials each, 256 tokens per output, temperature zero,
thinking disabled, one stream, no prefix-cache hits. Each type contains six
observations per configuration. A 32-token counting prompt warmed each route;
unlike the earlier compilation comparison, this run did not explicitly warm
every verifier width. All calibration and unsuccessful speculative work are
included in decode time. Reported values are medians, not best runs.

| Task type | Ordinary MLX tokens/s | DFlash + Metal tokens/s | Selected adaptive tokens/s | Median paired adaptive / ordinary |
| --- | ---: | ---: | ---: | ---: |
| Code | 17.46 | 21.16 | **22.47** | 1.26x |
| Math | 17.23 | 27.20 | **27.14** | 1.57x |
| Prose | **16.71** | 9.97 | 14.96 | 0.90x |

The paired ratio is computed per prompt and trial, then median-aggregated;
it is not the ratio of the two column medians. Selected-route ranges were
15.77–31.24 code, 23.31–29.80 math, and 13.51–20.76 prose tokens/s.

#### Per-task detail

Each cell below is the median of two trials. These finer-grained results are
important: a scientific explanation and an original story are not equally
predictable to this drafter.

| Task | Ordinary MLX tokens/s | DFlash + Metal tokens/s | Selected adaptive tokens/s |
| --- | ---: | ---: | ---: |
| Code: linked-list LRU cache | 18.49 | 30.18 | 29.78 |
| Code: JSONL validation and ranking | 17.81 | 20.58 | 22.47 |
| Code: threaded worker pool | 16.40 | 18.86 | 16.96 |
| Math: quadratic derivation | 17.77 | 29.42 | 29.05 |
| Math: Bayes calculation | 16.28 | 25.59 | 24.94 |
| Math: induction and telescoping | 16.37 | 26.99 | 26.26 |
| Prose: scientific explanation | 18.97 | 21.09 | 19.97 |
| Prose: original essay | 16.23 | 9.46 | 14.22 |
| Prose: original story | 15.84 | 9.97 | 14.27 |

The gate switched both essay trials at output token 21 and both story trials
at token 23, after eight speculative rounds. It did not switch the other
seven prompts. Essay and story median rates rose about 50% and 43% over
unconditional DFlash, respectively, but remained about 12% and 10% below
ordinary MLX. Worker-pool generation is a cautionary case: the adaptive route
needed 68 verification rounds versus 61, with little advantage over ordinary
decoding. Small differences among fast variants should not be treated as
established gains in this noisy environment.

For this particular equal-weight, 18-request workload, summed warm request
time (including prefill) was 287.46 seconds ordinary, 267.94 seconds DFlash,
and 238.14 seconds adaptive: a 17.2% reduction versus ordinary and 11.1% versus
unconditional DFlash. This is a measured workload mix, not a universal average.
Peak observed MLX allocation was 20.82 GB without expanded weights.

#### Generalization and correctness limits

- These are development prompts used to refine the gate, not a held-out eval.
  Two repeats do not provide a robust statistical confidence interval.
- Ordinary controls ranged from 15.10 to 20.69 tokens/s across the final run.
  Background workloads and thermals were not controlled; absolute rates from
  different research passes must not be compared as if only code changed.
- All broad-suite outputs hit the 256-token limit. They establish throughput
  for generated prefixes, not completion or correctness of full programs,
  proofs, essays, or stories. Thinking mode was disabled.
- The adaptive output matched ordinary-decoder token hashes in 6/18 cases
  (Bayes, induction, and essay in both trials); unconditional DFlash matched
  in 4/18. Each configuration's hash repeated across its two trials. Different
  batched arithmetic can alter close greedy choices; these measurements do
  not establish bitwise equivalence or unchanged task accuracy. The separate
  complete-copy experiment did match all tokens and pass its exact edit check.
- Long contexts, sampled real-model output quality, unseen prompts, concurrent
  users, sustained thermally controlled runs, and other Macs remain unqualified.
  Native sampled-distribution/cache tests are not a substitute for those evals.

**Initial research decision:** keep adaptive DFlash opt-in as the reproducible general research
baseline, retain ordinary MLX as the conservative choice for original prose,
and keep the 120.50-token/s copy path separate. Do not claim 100 tokens/s on
general prose, math, or code; do not make a new application-default change
from this evidence alone.

### User-selected branch baseline

The user subsequently requested making the selected configuration the default
on this branch for hands-on testing and opening a PR. `MLXLMBackend` now
constructs its automatically discovered DFlash decoder with `adaptive=True`,
while retaining the existing per-shape Metal qualification. No new opt-in
setting is required. This is a product-testing decision, not new evidence of
a universal speedup or unchanged task accuracy.

Scope remains Qwen3.8-27B on MLX with its installed, compatible auxiliary and
an output budget of at least 32 tokens. Missing auxiliary weights, unsupported
models, and shorter budgets retain ordinary decoding. `GEIST_MLX_DFLASH=off`
still forces ordinary MLX; explicit `on` still surfaces initialization errors.
The runner does not download anything. Copying, expanded weights, and compiled
recurrence stay off, and sampling/thinking defaults are unchanged.

Restart the loaded native runner from this branch to test the new default.
Initialization logs identify adaptive DFlash with qualified Metal kernels;
generation statistics report `implementation=mlx_dflash`, `adaptive=true`,
and `adaptation_policy=cumulative-eight-v2`, plus any `fallback_at_token`.
The benchmark harness retains independent options for historical ablations;
its bare invocation is not an application-default selection API.

Promotion verification: **151 native tests** (6.28 seconds) and **229 Docker
contract/service tests** (1.17 seconds) passed, including automatic default
activation and ordinary-path preservation. Two real `MLXLMBackend` requests
ran without DFlash environment overrides: the 128-token greedy essay fell
back at token 21; sampled code emitted 128 tokens without fallback. Both
reported adaptation enabled and 395 qualified wrappers; copying, expansion,
and compilation stayed disabled. These are activation smoke tests, not new
headline throughput measurements. See [default smoke evidence](../benchmarks/mlx/2026-09-05-lab/branch-default-smoke.json)
and the promotion section of [validation evidence](../benchmarks/mlx/2026-09-05-lab/validation.md).

Raw evidence:
[final repeated run](../benchmarks/mlx/2026-09-05-lab/general-adaptive-cumulative-v2-extended.json),
[derived summary](../benchmarks/mlx/2026-09-05-lab/general-summary.json).
The summary is derived from per-result configuration flags; the raw file's
legacy `median_dflash_tps` mixes configurations and must not be used for this
comparison.

### Verification

- Native MLX regression suite: **151 passed**, three existing Pydantic
  deprecation warnings, 6.80 seconds in the final run.
- Docker inference, runner, catalog, completion, settings and policy tests:
  **227 passed**, three existing Pydantic warnings.
- Targeted Ruff lint and format checks passed on 14 changed Python files.
  Project-configured mypy passed on 11 source files; the existing configuration
  suppresses errors in the llama architecture package, so this is not a strict
  type-check claim for that package.
- Isolated Docker and native API startup succeeded with disposable SQLite
  data. `/docs` and `/openapi.json` returned HTTP 200 on ports 5586 and 5587;
  observed startup/request logs contained no errors or tracebacks.
- Full `docker compose up -d --build`, `make run MLX_BACKEND=1`, and browser
  chat/settings flows were not run for this lab checkout. Ports 3000/5001 were
  already occupied by another worktree; no services were displaced, frontend
  dependencies installed, or unrelated UI treated as this branch's evidence.
  Native inference itself was exercised with the real checkpoints above.

Exact validation commands and isolated-smoke details are retained in
[validation evidence](../benchmarks/mlx/2026-09-05-lab/validation.md).
No dependencies changed. PR #356 and the original dirty worktree were not
modified or republished by this pass.

### Reproduction

Use the existing approved MLX environment and the same pinned local snapshots:

```sh
python scripts/benchmark_mlx_dflash.py \
  --weights-dir /absolute/path/to/target/snapshot \
  --drafter-dir /absolute/path/to/drafter/snapshot \
  --small-m --autotune --compile-sweep --warm-verifier-widths \
  --copy-window 0 --suite extended --max-tokens 256 --trials 2 \
  --output /tmp/general-baseline.json

python scripts/benchmark_mlx_dflash.py \
  --weights-dir /absolute/path/to/target/snapshot \
  --drafter-dir /absolute/path/to/drafter/snapshot \
  --small-m --autotune --adaptive-sweep --copy-window 0 \
  --suite extended --max-tokens 256 --trials 2 \
  --output /tmp/general-adaptive.json
```

Configuration order reverses on alternating trials; the ordinary decoder is
always first. System workloads and thermals were not isolated, so paired
controls and per-prompt variation matter more than small headline differences.
These short, greedy runs do not establish long-context, sampled-generation,
multi-user, or cross-hardware performance.

For implementation details, negative experiments, hardware bounds, and sources,
see [the laboratory report](mlx-100tps-lab.md) and
[the existing runtime log](mlx-dflash-runtime.md).
