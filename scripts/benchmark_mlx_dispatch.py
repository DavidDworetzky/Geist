"""Dependent real-weight screen for dispatch and mixed-accumulation experiments."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from agents.architectures.llama.qwen_kernel_lab import lab_matmul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--chain-length", type=int, default=8)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "native",
            "staged",
            "pad",
            "fp16_native",
            "relaxed",
            "fast",
            "half_acc",
            "chunked_half_acc",
        ],
    )
    parser.add_argument("--pad-rows", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, _ = load(args.weights_dir)
    groups = {}
    for _, layer in model.named_modules():
        if isinstance(layer, nn.QuantizedLinear) and layer.bits == 4 and layer.group_size == 64:
            k, n = layer.weight.shape[1] * 8, layer.weight.shape[0]
            if k % 512 == 0 and n >= 4096:
                groups.setdefault((k, n), []).append(layer)
    results = []
    for (k, n), layers in groups.items():
        layers = layers[:3]
        inputs = [
            mx.random.normal((1, args.rows, k), key=mx.random.key(i)).astype(mx.bfloat16)
            for i in range(len(layers))
        ]
        reference = [layer(x) for layer, x in zip(layers, inputs, strict=True)]
        mx.eval(reference, inputs)
        for variant in args.variants:
            outputs = [
                lab_matmul(layer, x, variant, pad_rows=args.pad_rows)
                for layer, x in zip(layers, inputs, strict=True)
            ]
            mx.eval(outputs)
            error = max(
                float(
                    (
                        mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))
                        / mx.maximum(mx.max(mx.abs(ref.astype(mx.float32))), 1e-9)
                    ).item()
                )
                for out, ref in zip(outputs, reference, strict=True)
            )
            samples = []
            for trial in range(args.trials + 1):
                started = time.perf_counter()
                y = lab_matmul(layers[0], inputs[0], variant, pad_rows=args.pad_rows)
                for step in range(1, args.chain_length):
                    i = step % len(layers)
                    y = lab_matmul(
                        layers[i],
                        inputs[i] + mx.tanh(y[..., :1]) * 1e-4,
                        variant,
                        pad_rows=args.pad_rows,
                    )
                mx.eval(y)
                if trial:
                    samples.append((time.perf_counter() - started) * 1000 / args.chain_length)
            row = {
                "k": k,
                "n": n,
                "rows": args.rows,
                "variant": variant,
                "pad_rows": args.pad_rows,
                "milliseconds": statistics.median(samples),
                "relative_max_error": error,
                "chain_length": args.chain_length,
                "precision_variant": variant
                in {"fp16_native", "half_acc", "chunked_half_acc", "affine_fold"},
            }
            results.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).write_text(
        json.dumps(
            {"device": mx.device_info(), "mlx_version": mx.__version__, "results": results},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
