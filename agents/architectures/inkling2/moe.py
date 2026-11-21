"""MoE switch-linear layers with NVFP4 per-expert global scale support on MLX 0.31+."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _gather_sort(x: mx.array, indices: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    *_, M = indices.shape
    flat = indices.flatten()
    order = mx.argsort(flat)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], flat[order], inv_order


def _scatter_unsort(
    x: mx.array, inv_order: mx.array, shape: tuple[int, ...] | None = None
) -> mx.array:
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    """Quantized expert linear layer supporting affine, NVFP4 (+ global_scale), and MXFP8."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        global_scale: bool = False,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        scale = math.sqrt(1.0 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None
        if global_scale:
            self.global_scale = mx.ones((num_experts,), dtype=mx.float32)
        else:
            self.global_scale = None
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))
        self.freeze()

    @property
    def input_dims(self) -> int:
        return int(self.scales.shape[2]) * self.group_size

    @property
    def output_dims(self) -> int:
        return int(self.weight.shape[1])

    @property
    def num_experts(self) -> int:
        return int(self.weight.shape[0])

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        out = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        global_scale = self.get("global_scale")
        if global_scale is not None:
            # MLX's native nvfp4 global_scale multiplies the accumulated dot
            # product by (global_scale / (448 * 6)). Applying that exact factor
            # after gather_qmm works on mlx==0.31.2 without upgrading MLX.
            scale = global_scale[indices] * (1.0 / (448.0 * 6.0))
            while scale.ndim < out.ndim:
                scale = mx.expand_dims(scale, -1)
            out = (out.astype(mx.float32) * scale).astype(out.dtype)
        if "bias" in self:
            out = out + mx.expand_dims(self["bias"][indices], -2)
        return out


class SwitchLinear(nn.Module):
    """Unquantized expert linear layer that can convert to QuantizedSwitchLinear."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self) -> int:
        return int(self.weight.shape[2])

    @property
    def output_dims(self) -> int:
        return int(self.weight.shape[1])

    @property
    def num_experts(self) -> int:
        return int(self.weight.shape[0])

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        out = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            out = out + mx.expand_dims(self["bias"][indices], -2)
        return out

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        global_scale: bool = False,
    ) -> QuantizedSwitchLinear:
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            bias=False,
            group_size=group_size,
            bits=bits,
            mode=mode,
            global_scale=global_scale,
        )
        if "bias" in self:
            ql.bias = self.bias
        return ql
