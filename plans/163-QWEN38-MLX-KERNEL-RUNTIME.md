# Plan: Qwen 3.8 MLX kernel runtime

## Goal

Raise Qwen 3.8 27B throughput and practical generation limits on Apple Silicon
without replacing Geist's local MLX runner with an external serving backend.
Measure Geist-owned generation and kernel ideas against the pinned `mlx-lm`
implementation. Ship only improvements that beat the control while keeping
correctness and fallback behavior explicit.

## Baseline and constraints

- Target artifact: `mlx-community/Qwen3.8-27B-4bit`, pinned by Geist.
- Target architecture: 64 dense layers arranged as 48 Gated DeltaNet layers and
  16 full-attention layers, with a 5,120-wide hidden state and 17,408-wide FFN.
- The installed snapshot occupies about 16.05 GB. Single-token decode is
  primarily bounded by unified-memory bandwidth because each token reads the
  model weights once.
- M2 Max has 400 GB/s advertised bandwidth, so the optimistic file-size
  roofline is about 24.9 token/s before recurrent state, KV cache, activations,
  kernel launches, and contention.
- Plain autoregressive fusion cannot cross that roofline. Material speedups
  above it require prefix reuse, batching, or verifying multiple predicted
  tokens per target-model weight read.

## Implementation

1. Prototype a Geist-owned MLX backend for Qwen 3.8 while retaining `mlx_lm` as
   the control. The first scheduler and projection-fusion prototype measured
   5-8% slower on sustained decode, so it is intentionally not shipped.
2. Evaluate compatible 4-bit projection fusion after model load:
   - Q/K/V for full-attention layers;
   - QKV/Z/A/B for Gated DeltaNet layers;
   - FFN gate/up projections.
   The experiment preserved output tokens but regressed throughput, so the
   normal per-projection path remains active.
3. Retain MLX-LM's pipelined generation loop and sampler path; it remained
   faster than the custom Python scheduler.
4. Retain a live hybrid recurrent/KV cache and reuse it only when the next
   tokenized request has the entire cached token sequence as an exact prefix.
   Fall back to a fresh cache for branches or mismatches.
5. Make the prefill chunk size configurable and benchmark realistic shapes.
   Keep the default conservative until measurements on the M2 Max select it.
6. Stream token text through the runner contract rather than buffering the
   entire local completion before emitting one event.
7. Expand the benchmark harness with a two-turn cache measurement and JSON
   results suitable for roofline comparisons. Add automated shape sweeps with
   the speculative verifier rather than overfitting plain single-token decode.
8. Raise the UI-configurable MLX output ceiling independently of the default,
   avoiding an accidental increase to online-provider requests.

## Follow-on kernel work

The active second phase implements an in-process DFlash 2 backend with the
immutable `incoai/Qwen3.8-27B-DFlash2` checkpoint at revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. It reuses MLX-LM target weights,
captures target hidden states at layers 5/19/33/47/61, and reconstructs rejected
DeltaNet state from recorded recurrence inputs. The kernel dispatch is per-model
and benchmark-gated. Validation compares accepted token output and timing against
the same loaded target, including short and full acceptance/rejection paths.


- Adopt a small-M quantized matmul kernel which dequantizes each weight group once
  and reuses it across 2-8 verification rows using `simdgroup_matrix` tiles.
- Load Qwen's separate DFlash 2 head as an optional immutable artifact and
  autotune verification width from measured target-forward cost and acceptance.
- Explore concurrent CPU/GPU prefill only after the single-device shape sweep;
  it increases aggregate compute but also competes for unified-memory bandwidth.
- Quantize full-attention KV only above a measured context threshold. The fixed
  Gated DeltaNet state should remain float32 unless an equivalence gate proves a
  lower-precision state is stable.

## Acceptance criteria

- Qwen 3.8 keeps the fastest measured implementation as its default; experimental
  schedulers must beat the pinned `mlx_lm` control before they are exposed.
- Exact-prefix cache reuse is covered by tests and reports reused token counts.
- Benchmarks report prompt and decode token/s, generated count, peak memory,
  implementation, and follow-up cache reuse.
- Focused runner tests pass, followed by native MLX generation and the relevant
  Geist runtime smoke loop.

## Measurements on the M2 Max 96 GB

- MLX 0.31.2 control: 20.66 token/s over a forced 256-token generation.
- Custom scheduler: 19.30 token/s; projection fusion did not recover the loss.
- MLX 0.32.2 with cache-aware Geist adapter: 21.02 token/s over 256 tokens,
  15.48 GB peak allocated memory.
- Two-turn correctness check: 39 of 68 prompt tokens reused on turn two; answer
  remained correct and the turn completed in 0.68 seconds.

## Implemented second phase

- In-process DFlash 2 drafting with the pinned auxiliary snapshot, 8-bit drafter
  weights, seven proposed tokens, and the original 4-bit target.
- Instance-scoped, BF16-operand, float32-accumulation Metal QMM for verification
  shapes. Per-shape native/QMM comparison selects split-K 2/4/8 or native fallback.
- Recorded DeltaNet inputs recover rejected recurrent state without rerunning
  the target projections. Exact-prefix reuse includes draft context; EOS,
  output limits, and cancellation have native regression tests.
- Automatic use when the matching auxiliary artifact is installed, explicit
  `GEIST_MLX_DFLASH=off` control, and ordinary decoding for budgets below 32.
- The actual runner streams decoded text and preserved 171 cached prompt tokens
  in a real two-turn check. The local output-budget ceiling is 131,072, not a
  claim that a 131,072-token run was benchmarked.

Three paired 256-token greedy trials produced median rates of 33.67 tok/s
(code, +59%), 37.68 tok/s (math, +79%), and 22.34 tok/s (prose, +7%). Math output
matched all baseline tokens; code and prose diverged at floating-point-sensitive
decisions. The maximum observed rate was 37.75 tok/s, not a theoretical bound.
See `docs/mlx-dflash-runtime.md` and `benchmarks/mlx/` for setup and raw evidence.

Remaining performance work is adaptive suppression when draft acceptance is
poor, more efficient recurrent verification, and longer-block verifier kernels.
No wider-block checkpoint behavior, batching, or new hardware speed is assumed
from these measurements. Full-app startup validation still requires permission
for startup's local `.env` read; direct Metal/runner tests do not read it.

## Third phase: challenge the resource tradeoffs

Profile draft, verification, acceptance, and recurrent rollback separately.
Compare expanded BF16 matrices against quantized verification: use spare memory
only if removing unpacking costs outweighs moving more bytes. Experiment with
longer draft/verify blocks behind explicit benchmark flags; the checkpoint was
trained with eight positions, so longer blocks require empirical acceptance and
correctness checks before any default changes. Retain the measured eight-position
configuration as the control throughout.

This phase added explicit experimental controls and native equivalence tests
for wider tiles, compressed packing, direct/shuffled MMA fragments, larger and
padded shared-memory stages, exact dequantization lookup tables, and two
lookahead selector objectives. None established a broad win and none is enabled
by automatic runtime selection. Profiling places most round time in target
verification. The kernel tuner now measures dependent projection chains instead
of overlapping independent calls, with a 12-call timing chain to reduce noise.

A 1,024-token paired trial sustained 35.25 tok/s on code and 37.38 on math;
all math tokens matched ordinary MLX. This is longer-run validation, not a new
peak. The best established short-run peak remains 37.75 tok/s. The report now
uses 14.42 GB of non-embedding language tensors for approximate weight-bandwidth
calculations, excluding vision tensors and full input-embedding-table scans.

Future architectural candidates are branch-aware verification inside a fixed
matrix-row budget, trained wider drafters, and counter-guided pipelined QMM.
Tree verification must explicitly handle branch-local DeltaNet/convolution
state, depth-correct RoPE and ancestor attention, and selected-path KV compaction.
Those are proposed follow-ons, not implemented functionality.
