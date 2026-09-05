# MLX throughput laboratory: aligned verification and memory tradeoffs

**Result: 120.50 tokens/s median across three complete synthetic code-edit runs
on M2 Max 96 GB.** Range 108.72–126.53; all 1,802 tokens in every run matched
the ordinary decoder. This is verified reuse of known text, not a claim of
100 tokens/s for novel chat. The path stays opt-in.

This branch starts at PR #356 (`eef0ae3`) and leaves that PR unchanged. After
the research pass, the user selected adaptive DFlash + qualified Metal as this
branch's application default. Copying, expanded weights, and compilation stay
explicit benchmark options. No backend migration, model download, dependency
change, or machine-wide performance setting was required.

Environment: MLX 0.32.2, MLX-LM 0.31.3; target
`mlx-community/Qwen3.8-27B-4bit` at
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`; auxiliary
`incoai/Qwen3.8-27B-DFlash2` at
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, quantized in memory to eight bits.

## What the hardware measurements say

On this Apple M2 Max 96 GB, ordinary Qwen3.8-27B Q4 decoding measured roughly
15–21 tokens/s across the research passes and changing system load. More output
tokens do not make that serial operation faster.
The useful objective is accepted new tokens divided by draft, verification,
acceptance, and state-recovery time combined.

The target's roughly 14.42 GB non-embedding weight stream has an idealized
36 ms read time at Apple's advertised 400 GB/s. That is a lower bound on one
weight sweep, not a prediction of achieved throughput. Eight-position drafting
would need nearly perfect acceptance and a much cheaper verifier to reach
100 tokens/s. Real verifier measurements, including host submission and GPU
completion but excluding drafting, demonstrate the gap:

| Verification rows | Ordinary verifier ms | Compiled recurrent layers ms |
| --- | ---: | ---: |
| 8 | 165.2 | 152.6 |
| 32 | 275.6 | 273.7 |
| 64 | 517.3 | 516.9 |
| 128 | 976.1 | 976.0 |

These are synthetic token-block shape probes at short context, three warm
trials per size. Their perfect-acceptance rates are not generation benchmarks.
An eight-row verifier at 165 ms cannot emit 100 new tokens/s even with a free,
perfect drafter. A longer trained drafter or another reliable proposal source
is necessary unless verifier latency changes radically.

## Independent implementation pieces

1. **Known-text proposals.** `NgramDraft` indexes token history with eight-token
   matches, extends matches backward to resolve repeated prefixes, and proposes
   a known continuation. Every proposed token must match the target verifier's
   greedy argmax. Bad copies fall back to ordinary DFlash with a four-round
   cooldown. The current experiment rejects nonzero-temperature use.
   The final lookup also remembers source continuity: after an accepted span,
   its next search considers the expected next source position even when that
   repetitive ngram was pruned from the bounded occurrence index. It still
   requires a real suffix match, and the target still verifies the proposal.
2. **Anchor-aligned shapes.** A proposal needs one extra row for the pending
   anchor. A 127-token proposal creates 128 rows; 128 proposals create 129.
   This distinction matters dramatically on the installed MLX/M2 kernels.
3. **Expanded MLP weights for wide blocks.** Keep the original Q4 weights for
   ordinary decoding and small-row verification. Additionally dequantize the
   192 MLP projections into BF16 in memory and use dense matmul for 64–256 rows.
   This costs about 34.2 GB extra, with explicit budget and GPU-working-set
   guards. It does not recover the original unquantized checkpoint: these are
   the same quantized values, expanded once. FP16 is a separate precision
   experiment, not the BF16 result.
4. **Compiled recurrent layers.** Explicit cache input/output arrays let MLX
   compile recurrent-layer operations while retaining the inputs needed for
   rollback. Compiled routing choices must stay fixed after tracing. This
   remains independently selectable, not required for copying or expansion.
5. **Per-request break-even fallback.** Measure four real native output tokens,
   then accumulate speculative tokens and time. After at least eight rounds,
   switch the remaining request to native steps if cumulative speculative
   throughput is below 85% of the measured native reference. Preserve target
   state and bounded deferred drafter context for follow-up prefix reuse.
   The first rolling-window policy was rejected because it switched useful
   code/math requests too readily. Both versions' data are retained.

The copy approach follows the general verified-copy principle described by
[CopySpec](https://arxiv.org/abs/2502.08923); this branch's small lookup
implementation was written locally rather than imported from another backend.

### The measured shape cliff

A separate paired shape run, with expanded weights resident for both paths:

| Rows including anchor | Q4 verifier ms | Expanded MLP verifier ms |
| --- | ---: | ---: |
| 64 | 524.3 | 458.0 |
| 65 | 784.8 | 759.0 |
| 128 | 1012.9 | 865.7 |
| 129 | 1311.6 | 1267.7 |

Thus the first important optimization was proposing *one fewer token*, not
asking for a longer output. Expanded weights help the aligned wide operations
but do little to rescue the misaligned ones. This is measured behavior of these
kernels, not a universal Apple GPU boundary.

## Initial end-to-end code-edit screen

The synthetic prompt asks to change only `DEFAULT_TIMEOUT = 30` to `45` and
reproduce a repetitive Python configuration file. It has 1,866 prompt tokens.
Each row below is a separate one-trial, 512-output-token experiment, not a
repeated median. All 512 tokens matched its ordinary-decoder control exactly.

| Copy proposal limit | Expanded MLP | Decode tokens/s | Peak MLX GB |
| --- | --- | ---: | ---: |
| 64 | No | 64.87 | 20.82 |
| 128 | No | 72.14 | 20.82 |
| 127 | No | 85.54 | 20.82 |
| 127 | BF16 | 96.26 | 54.58 |

The last row's paired control was 20.45 tokens/s. Total request time, including
prefill, fell from 38.15 to 18.42 seconds—about 2.07x, not the 4.71x decode-only
speedup. Loading, dequantizing the shadow weights, tuning, and initial warm-up
are excluded from these warm-request times. The 512-token budget truncates the
file, so it does not prove completion of the editing task.

These results do **not** establish 100 tokens/s general chat, nor universal
bit-identical decoding. Batch size and reduction order can alter close greedy
decisions even when the checkpoint and intended arithmetic are unchanged.
Exact-token comparisons must be recorded per workload and configuration.

The first complete-file screen exposed a retrieval issue: with the original
lookup, repeated code patterns discarded useful intermediate source positions.
It proposed 2,413 copy tokens but accepted only 1,584. The whole 1,802-token
response still matched the control, but decode dropped to 78.43 tokens/s
versus 38.27 for DFlash alone and 18.49 for ordinary MLX. This motivated source
continuity, rather than hiding the slower full-length result.

### Repeated complete-file result after source continuity

| Trial | Ordinary MLX tokens/s | Copy + aligned BF16 MLP tokens/s | Ordinary total s | Optimized total s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 18.38 | 120.50 | 112.29 | 29.29 |
| 2 | 18.40 | 108.72 | 112.32 | 32.43 |
| 3 | 20.12 | 126.53 | 104.09 | 30.85 |

Three complete outputs, each 1,802 tokens including EOS, matched the paired
control's SHA-256 `b63c28e453162eb4a783cff572784ab9043e1041186c499a42e5ec18c46f2701`.
Extracted code parsed successfully and matched the source with only the
requested timeout change; it was not merely semantically similar code.
The maximum output budget was 2,048, with natural EOS ending every run.

Median decode speed was 120.50 versus 18.40 tokens/s; median paired decode gain
was 6.29x. Median total request time was 30.85 versus 112.29 seconds, with
13–17 seconds of prefill included. These totals still exclude one-time loading,
weight expansion and tuning. Process peak MLX allocation reached 55.29 GB over
the three runs. Each request processed a fresh prefix: cached prompt tokens=0.

Source continuity reduced the run to 20 verification rounds: 15 copy rounds,
1,748 accepted copy tokens from 1,905 proposals, plus five ordinary DFlash
rounds. The target still evaluated every proposed position and corrected any
mismatch. No unverified tokens were counted, and no concurrency/batching rate
was substituted for single-stream generation.

Raw results are in
`benchmarks/mlx/2026-09-05-lab/copy-continuation-repeated.json`. The earlier
complete-file result and all initial screens are retained alongside it.

## Negative and inconclusive screens

- Padding eight input rows to sixteen did not beat the staged Q4 kernel. MLX's
  M2 dispatch already selects a matrix kernel at that size; padding does not
  unlock a new tensor unit.
- Relaxed/fast compiler math gave no substantial improvement in the tested
  matmuls. No fast-math mode was applied to attention masking or sampling.
- Half accumulation and half accumulation flushed every 64 K values produced
  mixed shape-specific results and change precision. Neither is a default.
- Moving affine quantization scale/bias outside each 64-value dot product
  changed dequantization rounding and was slower on every screened shape.
- The CPU Q4 path was thousands of milliseconds for one FFN projection, versus
  sub-millisecond staged GPU execution at eight rows. That run was interrupted.
  CPU dense and CPU/GPU output-row splits did not establish an overall win.
  A wide CPU/GPU run briefly overlapped the interrupted process; its timings
  are not used as clean performance evidence.
- A dense GPU MLP screen suggested a wide-row benefit; the full-target paired
  shape experiment above subsequently confirmed a smaller real verifier gain.
  Expanding weights for eight-row decode remains a loss, consistent with the
  earlier PR's small-row experiment.
- Xcode's `xctrace` was unavailable. Per-phase software fences were used for
  attribution; their synchronization overhead must not be summed into a claimed
  unfenced performance model. No hardware occupancy counters were collected.

### Ordinary-generation screen: not a general 100 tokens/s result

The first code/math/prose spot-check generated 256 tokens per prompt. Copying
did not activate on code or prose; math used one copy round and saved one
verification round. Without compilation, rates with copying enabled were
34.47 code, 34.51 math, and 20.29 prose tokens/s. The corresponding DFlash
controls with copying disabled were 34.26, 33.38, and 21.06 tokens/s.

Compilation produced 35.04 code and 35.94 math tokens/s with copying disabled,
but one prose run fell to 14.69 versus 21.06. That single-trial suite does not
establish a robust compile win: shape-specific first-use costs and system load
were not separated. All speculative variants had the same token hashes within
each prompt, but their hashes differed from ordinary decoding. The first
differences were indices 215 (code), 78 (math), and 173 (prose).

To select a general baseline, a separate extended suite uses three prompts per
task type, two trials each, and explicit warm-up of verification widths 1–8
for both target implementations. Copying and weight expansion are disabled.
This suite measures current DFlash/Metal against compiled recurrence on the
same loaded target. It is a representative throughput suite, not a broad
accuracy evaluation or a proof of performance at every context length.

The completed warm-width suite found compilation effectively tied: code
27.69 to 27.57, math 29.30 to 29.33, prose 11.77 to 11.87 tokens/s. Essays and
stories, unlike the scientific explanation, exposed severe low-acceptance
regressions. That prompted the adaptive fallback, not a compilation default.

The final repeated comparison selected **adaptive DFlash + qualified small-row
Metal, with copying/expanded weights/compilation off**, as the
mixed-workload research baseline, subsequently promoted to the branch default
at the user's request:

| Type, six observations each | Ordinary MLX tokens/s | DFlash + Metal tokens/s | Adaptive tokens/s |
| --- | ---: | ---: | ---: |
| Code | 17.46 | 21.16 | 22.47 |
| Math | 17.23 | 27.20 | 27.14 |
| Prose | 16.71 | 9.97 | 14.96 |

These are medians over three prompts per type and two trials, each 256 tokens,
greedy, thinking disabled, fresh prefixes. Absolute rates drifted between
passes; compare each with its contemporary controls. The gate helped essays
and stories substantially relative to unconditional DFlash, but ordinary
decoding still won on both. Only 6/18 adaptive outputs matched ordinary token
hashes, and these truncated, development-set outputs are not a task-accuracy
qualification. A generalized 100 tokens/s or universally faster default is
not established. Full per-task values, exact commands, validation limits,
negative decisions, and raw-result links are in the
[feature research log](feature-log.md).

## Reproduction and controls

Use the existing approved native MLX environment. No installation is implied:

```sh
python scripts/benchmark_mlx_dflash.py \
  --weights-dir /absolute/path/to/target/snapshot \
  --drafter-dir /absolute/path/to/drafter/snapshot \
  --small-m --autotune --copy-window 127 --expanded-mlp bf16 \
  --prompt-file benchmarks/mlx/2026-09-05-lab/copy-edit-prompt.txt \
  --max-tokens 2048 --trials 3 --output /tmp/copy-edit.json
```

`--copy-sweep 0 127` compares proposal sources on the same loaded target.
`--expanded-sweep` compares both routes with the shadow weights resident in
both cases. Configuration order reverses on alternating trials; the ordinary
decoder runs first in each pair. `--compile-verifier` is separate and cannot
be combined with a dynamic expanded-weight sweep.

The shape profiler includes one discarded warm-up and three samples per width.
Projection dispatch screening uses dependent chains over up to three real
weight matrices, numerical error checks, and a separate warm-up. Microbenchmarks
do not automatically qualify runtime changes. Other workloads, thermals, and
system memory pressure are not controlled hardware-laboratory conditions.

## Architecture implications and remaining work

- A trained longer DFlash/MTP proposal source is the main route to these wider
  matrices for novel prose. Increasing the block size on an eight-position
  checkpoint already failed in the earlier experiments.
- Tree verification could spend wider matrices on alternative branches, but
  this hybrid model needs branch-local DeltaNet and convolution state as well
  as ancestor attention masks. Flattening branches into an ordinary causal
  sequence is incorrect. It is research work, not an implemented capability.
- CPU/Neural Engine overlap competes for the same memory bandwidth. The ANE
  requires a different supported graph and quantization path; its advertised
  TOPS cannot simply be added to GPU decode throughput. No ANE runtime was
  installed or benchmarked here.
- These are M2 Max measurements only. RAM capacity alone does not predict
  throughput on a 512 GB Studio. Measure its actual SoC, GPU kernels and memory
  bandwidth before projecting a ceiling; 170 tokens/s is not established here.

Primary implementation/hardware references:
[MLX 0.32.2 quantized dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp),
[Apple M2 Max bandwidth](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/),
[MLX custom Metal kernels](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/custom_metal_kernels.rst),
[Core ML optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html).
