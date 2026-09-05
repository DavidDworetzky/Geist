"""Opt-in RAM-for-compute experiment for wide verification blocks."""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from agents.architectures.llama.qwen_small_m import SmallMLinear


class ExpandedLinear(nn.Module):
    def __init__(self, linear, *, min_rows: int, max_rows: int, half: bool = False):
        super().__init__()
        self.linear = linear
        quantized = linear.linear if isinstance(linear, SmallMLinear) else linear
        self.expanded_weight = mx.dequantize(
            quantized.weight,
            quantized.scales,
            quantized.biases,
            group_size=quantized.group_size,
            bits=quantized.bits,
        )
        if half:
            self.expanded_weight = self.expanded_weight.astype(mx.float16)
        mx.eval(self.expanded_weight)
        self.min_rows, self.max_rows = min_rows, max_rows
        self.enabled = True

    def __call__(self, x):
        rows = math.prod(x.shape[:-1])
        if not self.enabled or not self.min_rows <= rows <= self.max_rows:
            return self.linear(x)
        output = (x.astype(self.expanded_weight.dtype) @ self.expanded_weight.T).astype(x.dtype)
        quantized = self.linear.linear if isinstance(self.linear, SmallMLinear) else self.linear
        return output + quantized.bias if "bias" in quantized else output


def install_expanded_mlp(
    model, *, budget_gb: float, min_rows: int = 64, max_rows: int = 256, half: bool = False
) -> list[ExpandedLinear]:
    if budget_gb <= 0 or not 1 <= min_rows <= max_rows:
        raise ValueError("Expanded weights need a positive budget and valid row interval")
    candidates = []
    for name, layer in model.named_modules():
        if not name.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
            continue
        quantized = layer.linear if isinstance(layer, SmallMLinear) else layer
        if not isinstance(quantized, nn.QuantizedLinear) or quantized.mode != "affine":
            continue
        if quantized.scales.dtype != mx.bfloat16:
            continue
        size = quantized.weight.size * (32 // quantized.bits) * 2
        candidates.append((name, layer, size))
    needed = sum(size for _, _, size in candidates)
    recommended = mx.device_info()["max_recommended_working_set_size"]
    if needed > budget_gb * 1e9 or needed + mx.get_active_memory() > recommended * 0.85:
        raise ValueError(f"Expanded MLP needs {needed / 1e9:.2f} GB beyond current model memory")
    replacements = [
        (name, ExpandedLinear(layer, min_rows=min_rows, max_rows=max_rows, half=half))
        for name, layer, _ in candidates
    ]
    model.update_modules(tree_unflatten(replacements))
    return [layer for _, layer in replacements]
