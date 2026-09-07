# MLX 100 tok/s investigation

## Objective and controls

Try to raise single-stream Qwen3.8-27B throughput toward 100 tok/s on the
installed M2 Max 96 GB while retaining MLX and the pinned target checkpoint.
Start from PR #356 (`eef0ae3`). Keep this laboratory branch separate from the
published PR and do not change production defaults based on microbenchmarks.
No new packages, models, machine-wide performance settings, or services are
required for the initial experiments.

The same synthetic code/math/prose prompts, output budget, target weights,
sampling policy, and warm-paired protocol remain the main comparison. Record
tokens, prefix agreement, accepted tokens per round, total elapsed time and
decode time separately. Prompt-copy/repetitive workloads and aggregate batch
rates must be labeled separately; neither proves general 100 tok/s chat.

## First-principles budget

Current rounds emit approximately 5–7 tokens in 150–190 ms. At that acceptance,
100 tok/s requires a 50–70 ms round. The ~14.42 GB target weight stream alone
costs at least 36 ms at advertised 400 GB/s. We therefore need a much more
efficient verifier, substantially more accepted tokens, or both; launch-only
improvements and higher output limits cannot close this gap.

## Experiments, in order of information gained per cost

1. Instrument the actual verification graph, separating quantized projections,
   full attention, recurrent update, pointwise work, head, and host submission.
   Measure fenced attribution and unfenced end-to-end throughput separately.
2. Test dispatch surgery: pad small verification matmuls to native GEMM shapes;
   compare native, current staged kernels, and larger-tile/pipelined Metal
   kernels using dependent chains over actual weights. Test safe/relaxed math
   and FP16 operands explicitly, with numerical checks.
3. Test graph compilation of pure per-layer functions with explicit cache
   inputs/outputs, avoiding Python-state mutation inside compiled functions.
4. Investigate Apple-specific spare compute: CPU matrix units or Neural Engine
   can only help if transfer, synchronization, and shared-memory bandwidth costs
   do not outweigh their work. Do not assume their advertised TOPS apply to
   this quantized recurrent model.
5. Prototype long exact prompt/ngram proposals: reuse known text as an untrusted
   draft, verify every emitted token with the target, and measure separately
   on copy/edit/repetition and ordinary generation. Preserve hybrid rollback.
6. Consider branch-aware verification or chained drafting only after measuring
   verifier shape costs. Tree paths need ancestor masks, depth-based positions,
   branch-local recurrent/convolution state, and exact accepted-path compaction.
7. The expanded prose suite exposed low-acceptance regressions. Test an opt-in
   per-request break-even gate: emit four native calibration tokens, measure
   eight speculative rounds, and fall back to native single-token MLX when
   cumulative speculation is at least 15% slower than that reference. The
   first four-round rolling gate reacted to transient difficult spans and is
   rejected. Preserve the target cache
   and bounded delayed drafter context so follow-up cache reuse stays correct.

## Acceptance and validation

### Branch baseline promotion

After the repeated research pass, the user explicitly requested making adaptive
DFlash + qualified Metal the branch default and opening a PR for hands-on
testing. Wire `adaptive=True` in the existing `MLXLMBackend` initialization;
retain its shape qualification, artifact/compatibility fallback, under-32-token
native route, and explicit `GEIST_MLX_DFLASH=off` control. Do not enable copying,
expanded weights, or compiled recurrence. The standalone decoder and benchmark
retain explicit ablation controls so historical results remain reproducible.
Add a no-environment-override initialization regression test, then exercise the
real application backend with locally installed weights and document the PR's
exact default scope and known prose/quality limitations. No catalog, model
identity, download policy, API schema, or dependency changes are required.

- Native tests for new kernels, padding, sampling, rollback, cache reuse, EOS,
  cancellation and output limits as applicable.
- Repeated paired real-weight measurements before recommending defaults.
- Isolated Docker contract tests plus native MLX generation, per the Geist
  test-loop. Report application smoke checks that cannot be completed.
- Preserve every failed experiment in the result summary, without claiming
  exhaustive exploration or a hardware-independent ceiling.
- If a variant changes activation/dequantization arithmetic, report it as a
  precision variant, never as bit-identical optimization of the control.
