"""Separate unfenced verifier costs from deliberately fenced phase attribution."""

import argparse
import collections
import json
import operator
import statistics
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from agents.architectures.llama.qwen_compiled_verifier import CompiledQwenTarget
from agents.architectures.llama.qwen_small_m import install_small_m, tune_small_m
from agents.architectures.llama.qwen_speculative import QwenSpeculativeTarget, captured_delta


def fenced_forward(target, ids, cache):
    phases = collections.defaultdict(float)

    def measure(label, function, *args):
        mx.synchronize()
        started = time.perf_counter()
        value = function(*args)
        mx.eval(value)
        phases[label] += time.perf_counter() - started
        return value

    h = measure("embedding", target.body.embed_tokens, ids)
    fa_mask = create_attention_mask(h, cache[target.body.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[target.body.ssm_idx])
    target.verify_width = ids.shape[1]
    target.records = {}
    for index, (layer, state) in enumerate(zip(target.body.layers, cache, strict=True)):
        normalized = measure("normalization", layer.input_layernorm, h)
        if layer.is_linear:

            def recurrent(layer=layer, normalized=normalized, state=state, index=index):
                output, record = captured_delta(layer.linear_attn, normalized, ssm_mask, state)
                target.records[index] = record
                return output

            r = measure("gdn_with_projections", recurrent)
        else:
            r = measure("attention_with_projections", layer.self_attn, normalized, fa_mask, state)
        h = measure("residual", operator.add, h, r)
        normalized = measure("normalization", layer.post_attention_layernorm, h)
        r = measure("mlp", layer.mlp, normalized)
        h = measure("residual", operator.add, h, r)
    h = measure("normalization", target.body.norm, h)
    measure("lm_head", target.language_model.lm_head, h)
    return dict(phases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-dir",
        help="Local target checkpoint directory; never downloads weights.",
        required=True,
    )
    parser.add_argument(
        "--rows",
        help="Projection or verification row counts to measure.",
        nargs="+",
        type=int,
        default=[1, 8, 16, 32, 64],
    )
    parser.add_argument(
        "--trials", help="Measured trials after an unreported warm-up.", type=int, default=3
    )
    parser.add_argument(
        "--output", help="Write per-trial results as JSON to this path.", required=True
    )
    parser.add_argument(
        "--compile-verifier", help="Use the opt-in compiled target verifier.", action="store_true"
    )
    parser.add_argument(
        "--expanded-mlp", help="Enable the opt-in dense MLP weight experiment.", action="store_true"
    )
    args = parser.parse_args()
    if args.compile_verifier and args.expanded_mlp:
        parser.error("The expanded-weight A/B changes routes; compiled routes must stay fixed")
    model, _ = load(args.weights_dir)
    wrappers = install_small_m(model)
    tuning = tune_small_m(wrappers)
    expanded = []
    if args.expanded_mlp:
        from agents.architectures.llama.qwen_expanded_mlp import install_expanded_mlp

        expanded = install_expanded_mlp(model, budget_gb=36)
    target_type = CompiledQwenTarget if args.compile_verifier else QwenSpeculativeTarget
    target = target_type(model, (5, 19, 33, 47, 61))
    results = []
    configurations = [
        (width, mode) for width in args.rows for mode in ([False, True] if expanded else [False])
    ]
    for width, expanding in configurations:
        for wrapper in expanded:
            wrapper.enabled = expanding
        samples, host = [], []
        for trial in range(args.trials + 1):
            cache = model.make_cache()
            mx.eval(model(mx.array([[10, 20, 30, 40]]), cache=cache), [c.state for c in cache])
            ids = mx.arange(50, 50 + width).reshape(1, -1)
            mx.eval(ids)
            started = time.perf_counter()
            logits, hidden = target.forward(ids, cache, capture=True)
            submitted = time.perf_counter()
            mx.eval(logits, hidden, [c.state for c in cache])
            finished = time.perf_counter()
            if trial:
                samples.append(finished - started)
                host.append(submitted - started)
        row = {
            "rows": width,
            "expanded": expanding,
            "verify_ms": statistics.median(samples) * 1000,
            "graph_build_ms": statistics.median(host) * 1000,
            "perfect_target_only_tps": width / statistics.median(samples),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    cache = model.make_cache()
    mx.eval(model(mx.array([[10, 20, 30, 40]]), cache=cache), [c.state for c in cache])
    phases = fenced_forward(target, mx.arange(50, 58).reshape(1, -1), cache)
    print(json.dumps({"fenced_8row_seconds": phases}), flush=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "device": mx.device_info(),
                "compiled": args.compile_verifier,
                "expanded_mlp": args.expanded_mlp,
                "results": results,
                "fenced_8row_seconds": phases,
                "fenced_path": "uncompiled; per-phase synchronization overhead included",
                "tuning": tuning,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
