"""Test CPU matrix-unit offload within MLX on one real FFN projection."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import load

from agents.architectures.llama.qwen_small_m import small_m_matmul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[8, 256])
    args = parser.parse_args()
    model, _ = load(args.weights_dir)
    layer = model.language_model.model.layers[0].mlp.gate_proj
    k, n = layer.weight.shape[1] * 8, layer.weight.shape[0]
    cpu = mx.new_stream(mx.cpu)
    gpu = mx.new_stream(mx.gpu)
    dense_bf16 = mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=64, bits=4)
    dense_fp16 = dense_bf16.astype(mx.float16)
    dense = dense_bf16.astype(mx.float32)
    mx.eval(dense, dense_fp16, dense_bf16)
    results = []
    for rows in args.rows:
        x = mx.random.normal((rows, k), key=mx.random.key(11)).astype(mx.bfloat16)
        mx.eval(x)
        reference = layer(x)
        mx.eval(reference)
        for variant in (
            "gpu_q4",
            "gpu_staged",
            "gpu_dense_bf16",
            "gpu_dense_fp16",
            "cpu_q4",
            "cpu_dense",
            "cpu10_gpu90",
            "cpu25_gpu75",
        ):
            if variant == "gpu_staged" and rows > 32:
                continue
            if variant == "cpu_q4" and rows > 8:
                continue

            def project(values, variant=variant):
                if variant == "gpu_q4":
                    return layer(values)
                if variant == "gpu_staged":
                    return small_m_matmul(layer, values)
                if variant == "gpu_dense_bf16":
                    return values @ dense_bf16.T
                if variant == "gpu_dense_fp16":
                    return (values.astype(mx.float16) @ dense_fp16.T).astype(mx.bfloat16)
                if variant == "cpu_q4":
                    return mx.quantized_matmul(
                        values,
                        layer.weight,
                        layer.scales,
                        layer.biases,
                        transpose=True,
                        group_size=64,
                        bits=4,
                        stream=cpu,
                    )
                if variant == "cpu_dense":
                    return mx.matmul(
                        values.astype(mx.float32, stream=cpu), dense.T, stream=cpu
                    ).astype(mx.bfloat16, stream=cpu)
                count = (n // (10 if variant == "cpu10_gpu90" else 4) // 64) * 64
                a = mx.matmul(
                    values.astype(mx.float32, stream=cpu), dense[:count].T, stream=cpu
                ).astype(mx.bfloat16, stream=cpu)
                b = mx.quantized_matmul(
                    values,
                    layer.weight[count:],
                    layer.scales[count:],
                    layer.biases[count:],
                    transpose=True,
                    group_size=64,
                    bits=4,
                    stream=gpu,
                )
                return mx.concatenate([a, b], axis=-1)

            samples = []
            error = None
            try:
                output = project(x)
                mx.eval(output)
                error = float(
                    (
                        mx.max(mx.abs(output.astype(mx.float32) - reference.astype(mx.float32)))
                        / mx.max(mx.abs(reference.astype(mx.float32)))
                    ).item()
                )
                for trial in range(4):
                    started = time.perf_counter()
                    output = project(x)
                    mx.eval(output)
                    if trial:
                        samples.append(1000 * (time.perf_counter() - started))
                row = {
                    "rows": rows,
                    "variant": variant,
                    "milliseconds": statistics.median(samples),
                    "relative_max_error": error,
                }
            except (ValueError, RuntimeError) as exc:
                row = {"rows": rows, "variant": variant, "unsupported": str(exc)}
            results.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).write_text(
        json.dumps({"device": mx.device_info(), "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
