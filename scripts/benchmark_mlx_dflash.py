#!/usr/bin/env python3
"""Paired warm baseline/DFlash benchmark using the same loaded MLX target."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from agents.architectures.llama.dflash_backend import DFlashDecoder, load_drafter
from agents.architectures.llama.mlx_lm_backend import _configure_thread_local_generation_stream
from agents.architectures.llama.qwen_small_m import (
    install_small_m,
    pack_small_m_weights,
    tune_small_m,
)


PROMPTS = {
    "code": "Write a complete Python LRU cache class using a doubly linked list and a dictionary. Include get, put, eviction, and a small test suite. Explain the time complexity.",
    "math": "Derive the quadratic formula from first principles by completing the square. Then solve three examples, showing each arithmetic step and checking the roots.",
    "chat": "Explain why the sky is blue and sunsets are red. Cover Rayleigh scattering, wavelengths, the atmosphere, and why clouds look white. Give a detailed explanation.",
    "code_jsonl": "Write a Python command-line program that reads a JSONL file, validates each record has an id and a numeric score, reports malformed lines with line numbers, and writes the ten highest scoring records. Use only the standard library. Include tests and explain how ties are handled.",
    "code_threads": "Implement a bounded Python worker pool using threading and queue from the standard library. Support graceful shutdown, returning results, and propagating worker exceptions. Include a usage example and explain how deadlocks are prevented.",
    "math_bayes": "A factory uses three machines producing 20%, 30%, and 50% of its items. Their defect rates are 1%, 2%, and 4%. Derive the probability that a defective item came from each machine using Bayes' theorem. Show every intermediate calculation, verify the probabilities sum to one, and explain the distinction between the conditional probabilities.",
    "math_induction": "Prove by induction that the sum of the first n cubes equals the square of n(n+1)/2. Expand the induction step explicitly. Then derive the same identity using a telescoping sum and compare the two proofs. Work through n=5 as a numerical check.",
    "chat_essay": "Write a thoughtful essay about the difference between being busy and making meaningful progress. Use concrete everyday examples, consider an objection to your argument, and finish with practical habits. Avoid bullet lists and motivational slogans.",
    "chat_story": "Write an original short story about a librarian who finds handwritten notes in books returned by someone who has not visited for thirty years. Use restrained prose, realistic dialogue, and a resolution that is surprising but not supernatural.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--drafter-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--prompts", nargs="+", choices=list(PROMPTS), default=["code", "math", "chat"]
    )
    parser.add_argument("--suite", choices=("standard", "extended"), default="standard")
    parser.add_argument("--caps", nargs="+", type=int, default=[7])
    parser.add_argument("--small-m", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument(
        "--lab-variant",
        choices=(
            "pad",
            "fp16_native",
            "relaxed",
            "fast",
            "half_acc",
            "chunked_half_acc",
            "affine_fold",
        ),
    )
    parser.add_argument("--pad-rows", type=int, default=16)
    parser.add_argument("--compile-verifier", action="store_true")
    parser.add_argument("--compile-sweep", action="store_true")
    parser.add_argument("--warm-verifier-widths", action="store_true")
    parser.add_argument("--copy-window", type=int, default=0)
    parser.add_argument("--copy-sweep", nargs="+", type=int)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--adaptive-sweep", action="store_true")
    parser.add_argument("--expanded-mlp", choices=("bf16", "fp16"))
    parser.add_argument("--expanded-budget-gb", type=float, default=36)
    parser.add_argument("--expanded-min-rows", type=int, default=64)
    parser.add_argument("--expanded-max-rows", type=int, default=256)
    parser.add_argument("--expanded-sweep", action="store_true")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--wide-kernel", action="store_true")
    parser.add_argument("--direct-fragment", action="store_true")
    parser.add_argument("--shuffle-fragments", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--selector-sweep", nargs="+", choices=("local", "prefix", "sequence"), default=["local"]
    )
    parser.add_argument("--experimental-block-size", type=int, choices=(8, 12, 16, 24, 32))
    parser.add_argument("--block-sweep", nargs="+", type=int, choices=(8, 12, 16, 24, 32))
    parser.add_argument("--split-k", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--half-operands", action="store_true")
    parser.add_argument("--column-tiles", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--stage-stride", type=int, choices=(8, 9, 10, 12, 16), default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--drafter-bits", type=int, choices=(0, 4, 8), default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.suite == "extended":
        args.prompts = list(PROMPTS)
    if args.lab_variant and args.autotune:
        parser.error("Laboratory variants require separate qualification; disable --autotune")
    if args.expanded_mlp and args.lab_variant:
        parser.error("Qualify expanded MLP and laboratory kernels separately")
    if args.expanded_sweep and (
        not args.expanded_mlp or args.compile_verifier or args.compile_sweep
    ):
        parser.error("Expanded A/B needs --expanded-mlp and an uncompiled verifier")
    copy_windows = args.copy_sweep or [args.copy_window]
    if any(not 0 <= window <= 128 for window in copy_windows):
        parser.error("Copy windows must be in 0..128")
    if args.prompt_file:
        PROMPTS["custom"] = args.prompt_file.read_text()
        args.prompts = ["custom"]
    _configure_thread_local_generation_stream()
    model, tokenizer = load(args.weights_dir)
    drafter = load_drafter(args.drafter_dir, model, args.drafter_bits)
    wrappers = install_small_m(model) if args.small_m else []
    # The shared embedding/head are already routed through the target wrappers.
    drafter.bind(model)
    wrappers += install_small_m(drafter) if args.small_m else []
    if args.lab_variant:
        if args.small_m:
            parser.error("Select either --small-m or --lab-variant")
        from agents.architectures.llama.qwen_kernel_lab import install_lab

        wrappers = install_lab(model, args.lab_variant, args.pad_rows)
        drafter.bind(model)
        wrappers += install_lab(drafter, args.lab_variant, args.pad_rows)
    for wrapper in wrappers:
        wrapper.split_k = args.split_k
        wrapper.half_operands = args.half_operands
        wrapper.column_tiles = args.column_tiles
        wrapper.stage_stride = args.stage_stride
        wrapper.max_rows = 32 if args.wide_kernel else 8
        wrapper.direct_fragment = args.direct_fragment or args.shuffle_fragments
        wrapper.shuffle_fragments = args.shuffle_fragments
        if args.packed:
            wrapper.packed_weights = pack_small_m_weights(wrapper.linear)
            mx.eval(wrapper.packed_weights)
    tuning = tune_small_m(wrappers) if args.autotune else []
    if tuning:
        print(json.dumps({"tuning": tuning}), flush=True)
    expanded = []
    if args.expanded_mlp:
        from agents.architectures.llama.qwen_expanded_mlp import install_expanded_mlp

        expanded = install_expanded_mlp(
            model,
            budget_gb=args.expanded_budget_gb,
            min_rows=args.expanded_min_rows,
            max_rows=args.expanded_max_rows,
            half=args.expanded_mlp == "fp16",
        )
        wrappers += expanded
        print(
            json.dumps(
                {"expanded_mlp_count": len(expanded), "active_gb": mx.get_active_memory() / 1e9}
            ),
            flush=True,
        )
    decoder = DFlashDecoder(
        model,
        tokenizer,
        drafter,
        profile=args.profile,
        experimental_block_size=args.experimental_block_size,
        copy_window=args.copy_window,
    )
    targets = {False: decoder.target}
    if args.compile_verifier or args.compile_sweep:
        from agents.architectures.llama.qwen_compiled_verifier import CompiledQwenTarget

        targets[True] = CompiledQwenTarget(model, drafter.config.target_layer_ids)
    if any(cap < 1 or cap >= decoder.block_size for cap in args.caps):
        parser.error("Every draft cap must be smaller than the block size")
    configurations = (
        [(size, size - 1) for size in args.block_sweep]
        if args.block_sweep
        else [(decoder.block_size, cap) for cap in args.caps]
    )
    configurations = [
        (size, cap, strategy, window, expand, compiled, adaptive)
        for size, cap in configurations
        for strategy in args.selector_sweep
        for window in copy_windows
        for expand in ([False, True] if args.expanded_sweep else [bool(expanded)])
        for compiled in ([False, True] if args.compile_sweep else [args.compile_verifier])
        for adaptive in ([False, True] if args.adaptive_sweep else [args.adaptive])
    ]
    results = []

    def run_baseline(ids, count):
        for wrapper in wrappers:
            wrapper.enabled = False
        tokens = []
        started = time.perf_counter()
        first = None
        for response in stream_generate(
            model,
            tokenizer,
            ids,
            max_tokens=count,
            sampler=make_sampler(args.temperature, top_p=args.top_p, top_k=args.top_k),
        ):
            if first is None:
                first = time.perf_counter()
            tokens.append(response.token)
        mx.synchronize()
        finished = time.perf_counter()
        return tokens, {
            "generation_tokens": len(tokens),
            "generation_tps": (len(tokens) - 1) / max(finished - first, 1e-9),
            "elapsed_seconds": finished - started,
            "prefill_seconds": first - started,
            "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "finish_reason": response.finish_reason,
        }

    def run_dflash(ids, count, cap, block_size, strategy, window, expand, compiled, adaptive):
        for wrapper in wrappers:
            wrapper.enabled = True
        for wrapper in expanded:
            wrapper.enabled = expand
        decoder.copy_window = window
        decoder.adaptive = adaptive
        decoder.target = targets[compiled]
        decoder.max_draft = cap
        decoder.block_size = block_size
        drafter.greedy_strategy = strategy
        tokens = list(
            decoder.generate(
                ids,
                max_tokens=count,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        )
        return tokens, dict(decoder.last_stats)

    warm = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Count from one to fifty, writing every number."}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("Warming baseline and DFlash...", flush=True)
    run_baseline(warm, 32)
    for block_size, cap, strategy, window, expand, compiled, adaptive in configurations:
        run_dflash(warm, 32, cap, block_size, strategy, window, expand, compiled, adaptive)
    width_warmup_seconds = 0.0
    if args.warm_verifier_widths:
        began = time.perf_counter()
        for target in targets.values():
            for width in range(1, decoder.block_size + 1):
                cache = target.make_cache()
                mx.eval(target.forward(mx.array([[10, 20, 30, 40]]), cache))
                logits, hidden = target.forward(
                    mx.arange(50, 50 + width)[None], cache, capture=True
                )
                mx.eval(logits, hidden, [state.state for state in cache])
                target.records = {}
        width_warmup_seconds = time.perf_counter() - began
        print(json.dumps({"width_warmup_seconds": width_warmup_seconds}), flush=True)
    for trial in range(args.trials):
        for name in args.prompts:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPTS[name]}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            baseline, baseline_stats = run_baseline(ids, args.max_tokens)
            results.append({"trial": trial, "prompt": name, "mode": "baseline", **baseline_stats})
            print(json.dumps(results[-1]), flush=True)
            ordered = configurations if trial % 2 == 0 else list(reversed(configurations))
            for block_size, cap, strategy, window, expand, compiled, adaptive in ordered:
                tokens, stats = run_dflash(
                    ids,
                    args.max_tokens,
                    cap,
                    block_size,
                    strategy,
                    window,
                    expand,
                    compiled,
                    adaptive,
                )
                first_diff = next(
                    (
                        i
                        for i, pair in enumerate(zip(baseline, tokens, strict=False))
                        if pair[0] != pair[1]
                    ),
                    None,
                )
                result = {
                    "trial": trial,
                    "prompt": name,
                    "mode": "dflash",
                    "cap": cap,
                    "selector_strategy": strategy,
                    "block_size": decoder.block_size,
                    "copy_window": window,
                    "expanded_mlp": args.expanded_mlp if expand else None,
                    "compile_verifier": compiled,
                    "small_m": args.small_m,
                    "packed": args.packed,
                    "wide_kernel": args.wide_kernel,
                    "direct_fragment": args.direct_fragment,
                    "shuffle_fragments": args.shuffle_fragments,
                    "split_k": args.split_k,
                    "half_operands": args.half_operands,
                    **stats,
                    "column_tiles": args.column_tiles,
                    "stage_stride": args.stage_stride,
                    "speedup": stats["generation_tps"] / baseline_stats["generation_tps"],
                    "tokens_identical": baseline == tokens,
                    "first_difference": first_diff,
                    "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    "text": tokenizer.decode(tokens),
                    "finish_reason": "stop" if tokens[-1] in tokenizer.eos_token_ids else "length",
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    report = {
        "hardware": platform.platform(),
        "device": mx.device_info(),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "mlx_version": mx.__version__,
        "drafter_bits": args.drafter_bits,
        "tuning": tuning,
        "results": results,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "lab_variant": args.lab_variant,
        "compile_verifier": args.compile_verifier,
        "compile_sweep": args.compile_sweep,
        "copy_window": args.copy_window,
        "copy_sweep": copy_windows,
        "adaptive": args.adaptive,
        "adaptive_sweep": args.adaptive_sweep,
        "expanded_sweep": args.expanded_sweep,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "suite": args.suite,
        "warm_verifier_widths": args.warm_verifier_widths,
        "width_warmup_seconds": width_warmup_seconds,
        "prompt_sha256": {
            name: hashlib.sha256(PROMPTS[name].encode()).hexdigest() for name in args.prompts
        },
        "expanded_mlp": args.expanded_mlp,
        "expanded_min_rows": args.expanded_min_rows,
        "expanded_max_rows": args.expanded_max_rows,
        "median_dflash_tps": statistics.median(
            r["generation_tps"] for r in results if r["mode"] == "dflash"
        ),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
