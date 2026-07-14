#!/usr/bin/env python3
"""Benchmark one Geist MLX Llama implementation and emit machine-readable JSON."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.architectures.base_runner import GenerationConfig
from agents.architectures.mlx_llama_runner import MLXLlamaRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("manual", "mlx_lm"), required=True)
    parser.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    parser.add_argument("--system-prompt", default="You are a concise helpful assistant.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    hardware = {
        "machine": platform.machine(),
        "platform": platform.platform(),
    }

    runner = MLXLlamaRunner()
    load_started = time.perf_counter()
    runner.load(
        args.model_id,
        {
            "implementation": args.implementation,
            "weights_dir": args.weights_dir,
        },
    )
    load_seconds = time.perf_counter() - load_started

    config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generation_started = time.perf_counter()
    messages = runner.complete(args.system_prompt, args.prompt, config)
    wall_seconds = time.perf_counter() - generation_started
    stats = dict(getattr(runner.llama, "last_stats", {}))

    result = {
        "implementation": args.implementation,
        "model_id": args.model_id,
        "weights_dir": args.weights_dir,
        "hardware": hardware,
        "load_seconds": load_seconds,
        "generation_wall_seconds": wall_seconds,
        **stats,
        "text": messages[-1]["content"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
