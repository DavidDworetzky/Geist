#!/usr/bin/env python3
"""Measure warm small-M kernels across the actual target's quantized shapes."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import load

from agents.architectures.llama.qwen_small_m import (
    install_small_m,
    make_dequant_table,
    pack_small_m_weights,
    small_m_matmul,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--chain-length", type=int, default=12)
    parser.add_argument("--independent", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--half-operands", action="store_true")
    parser.add_argument("--column-tiles", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--lookup", action="store_true")
    parser.add_argument("--direct-fragment", action="store_true")
    parser.add_argument("--shuffle-fragments", action="store_true")
    parser.add_argument("--group-batch", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--stage-stride", type=int, choices=(8, 9, 10, 12, 16), default=8)
    parser.add_argument("--stage-sweep", nargs="+", type=int, choices=(8, 9, 10, 12, 16))
    parser.add_argument("--split-k", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    args = parser.parse_args()
    if args.chain_length <= 0 or args.trials <= 0:
        parser.error("Chain length and trials must be positive")
    model, _ = load(args.weights_dir)
    groups = {}
    for wrapper in install_small_m(model):
        layer = wrapper.linear
        k = layer.weight.shape[1] * (32 // layer.bits)
        groups.setdefault((k, layer.weight.shape[0], layer.bits), []).append(layer)
    results = []
    for (k, n, bits), layers in groups.items():
        layers = layers[:3]
        inputs = [
            mx.random.normal((1, args.rows, k), key=mx.random.key(11 + index)).astype(mx.bfloat16)
            for index in range(len(layers))
        ]
        mx.eval(inputs)
        reference = [layer(x) for layer, x in zip(layers, inputs, strict=True)]
        mx.eval(reference)
        tables = (
            [make_dequant_table(layer) for layer in layers] if args.lookup else [None] * len(layers)
        )
        if args.lookup:
            mx.eval(tables)
        packed = (
            [pack_small_m_weights(layer) for layer in layers]
            if args.packed
            else [None] * len(layers)
        )
        if args.packed:
            mx.eval(packed)
        dense = (
            [
                mx.dequantize(
                    layer.weight,
                    layer.scales,
                    layer.biases,
                    group_size=layer.group_size,
                    bits=layer.bits,
                )
                for layer in layers
            ]
            if args.expanded
            else []
        )
        mx.eval(dense)
        layouts = [-1, 0, 8] if args.expanded else args.split_k
        configurations = [
            (split_k, stride)
            for split_k in layouts
            for stride in (args.stage_sweep or [args.stage_stride])
            if split_k > 0 or stride == (args.stage_sweep or [args.stage_stride])[0]
        ]
        for split_k, stage_stride in configurations:
            if split_k > 0 and (
                k % (split_k * args.group_batch * 64)
                or split_k
                * args.column_tiles
                * (args.group_batch * 128 * stage_stride + ((args.rows + 7) // 8) * 256)
                > 32768
            ):
                continue

            def project(
                index: int,
                x: mx.array,
                layers=layers,
                split_k=split_k,
                packed=packed,
                stage_stride=stage_stride,
                tables=tables,
                dense=dense,
            ) -> mx.array:
                layer = layers[index]
                if split_k > 0:
                    return small_m_matmul(
                        layer,
                        x,
                        split_k,
                        half_operands=args.half_operands,
                        column_tiles=args.column_tiles,
                        packed_weights=packed[index],
                        direct_fragment=args.direct_fragment or args.shuffle_fragments,
                        shuffle_fragments=args.shuffle_fragments,
                        group_batch=args.group_batch,
                        stage_stride=stage_stride,
                        dequant_table=tables[index],
                    )
                if split_k == -1:
                    out = x @ dense[index].T
                    return out + layer.bias if "bias" in layer else out
                return layer(x)

            outputs = [project(index, x) for index, x in enumerate(inputs)]
            mx.eval(outputs)
            errors = [
                mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))
                / mx.maximum(mx.max(mx.abs(ref.astype(mx.float32))), 1e-9)
                for out, ref in zip(outputs, reference, strict=True)
            ]
            error = float(mx.stack(errors).max().item())
            timings = []
            for trial in range(args.trials + 1):
                started = time.perf_counter()
                if args.independent:
                    outputs = [project(index, x) for index, x in enumerate(inputs)]
                    mx.eval(outputs)
                    calls = len(layers)
                else:
                    output = project(0, inputs[0])
                    for step in range(1, args.chain_length):
                        index = step % len(layers)
                        output = project(index, inputs[index] + mx.tanh(output[..., :1]) * 1e-4)
                    mx.eval(output)
                    calls = args.chain_length
                if trial:
                    timings.append((time.perf_counter() - started) / calls)
            result = {
                "k": k,
                "n": n,
                "bits": bits,
                "layers": len(layers),
                "split_k": split_k,
                "rows": args.rows,
                "dependent_chain": not args.independent,
                "chain_length": calls,
                "expanded_bf16": split_k == -1,
                "packed": args.packed,
                "lookup": args.lookup,
                "direct_fragment": args.direct_fragment,
                "shuffle_fragments": args.shuffle_fragments,
                "group_batch": args.group_batch,
                "stage_stride": stage_stride,
                "milliseconds": 1000 * statistics.median(timings),
                "half_operands": args.half_operands,
                "column_tiles": args.column_tiles,
                "relative_max_error": error,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
