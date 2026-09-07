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
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--drafter-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=list(PROMPTS))
    parser.add_argument("--caps", nargs="+", type=int, default=[7])
    parser.add_argument("--small-m", action="store_true")
    parser.add_argument("--autotune", action="store_true")
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
    _configure_thread_local_generation_stream()
    model, tokenizer = load(args.weights_dir)
    drafter = load_drafter(args.drafter_dir, model, args.drafter_bits)
    wrappers = install_small_m(model) if args.small_m else []
    # The shared embedding/head are already routed through the target wrappers.
    drafter.bind(model)
    wrappers += install_small_m(drafter) if args.small_m else []
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
    decoder = DFlashDecoder(
        model,
        tokenizer,
        drafter,
        profile=args.profile,
        experimental_block_size=args.experimental_block_size,
    )
    if any(cap < 1 or cap >= decoder.block_size for cap in args.caps):
        parser.error("Every draft cap must be smaller than the block size")
    configurations = (
        [(size, size - 1) for size in args.block_sweep]
        if args.block_sweep
        else [(decoder.block_size, cap) for cap in args.caps]
    )
    configurations = [
        (size, cap, strategy) for size, cap in configurations for strategy in args.selector_sweep
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
        }

    def run_dflash(ids, count, cap, block_size, strategy):
        for wrapper in wrappers:
            wrapper.enabled = True
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
    for block_size, cap, strategy in configurations:
        run_dflash(warm, 32, cap, block_size, strategy)
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
            for block_size, cap, strategy in configurations:
                tokens, stats = run_dflash(ids, args.max_tokens, cap, block_size, strategy)
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
        "median_dflash_tps": statistics.median(
            r["generation_tps"] for r in results if r["mode"] == "dflash"
        ),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
