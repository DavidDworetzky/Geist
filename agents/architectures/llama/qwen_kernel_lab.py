"""Explicit-only dispatch and accumulation experiments; never auto-installed."""

import math
from functools import cache

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from agents.architectures.llama.qwen_small_m import _SRC, _UNPACK, small_m_matmul


@cache
def experimental_kernel(bits: int, math_mode: str, accumulator: str):
    source = _SRC.replace("__UNPACK__", _UNPACK[bits])
    inputs = ["x", "w", "sc", "bi"]
    if accumulator == "affine_fold":
        inputs.append("xsum")
        for word in ("p0", "p1"):
            source = source.replace(
                f"(T)(bfloat16_t)((float)(({word} >> (4 * t)) & 15u) * s + bb)",
                f"(T)(({word} >> (4 * t)) & 15u)",
            )
        source = source.replace(
            "int kbeg = (int)sg * KPS;",
            """
            simdgroup_matrix<float, 8, 8> totals[TC * RT];
            for (int ct = 0; ct < TC * RT; ++ct)
                totals[ct] = simdgroup_matrix<float, 8, 8>(0);
            int qid = lane / 4;
            int fm = (qid & 4) + ((lane / 2) % 4);
            int fn = (qid & 2) * 2 + (lane % 2) * 2;
            int kbeg = (int)sg * KPS;
        """,
        )
        boundary = "        simdgroup_barrier(mem_flags::mem_threadgroup);\n    }"
        source = source.replace(
            boundary,
            """
            int group = (kbeg + kk) / 64;
            for (int rt = 0; rt < RT; ++rt) {
                float sx = xsum[(rt * 8 + fm) * (K / 64) + group];
                for (int ct = 0; ct < TC; ++ct) {
                    for (int e = 0; e < 2; ++e) {
                        int column = n0 + ct * 8 + fn + e;
                        float scale = float(sc[column * (K / 64) + group]);
                        float bias = float(bi[column * (K / 64) + group]);
                        totals[rt * TC + ct].thread_elements()[e] +=
                            C[rt * TC + ct].thread_elements()[e] * scale + sx * bias;
                    }
                    C[rt * TC + ct] = simdgroup_matrix<float, 8, 8>(0);
                }
            }
        """
            + boundary,
        )
        source = source.replace("simdgroup_store(C[ct], red", "simdgroup_store(totals[ct], red")
    elif accumulator != "float32":
        source = source.replace("simdgroup_matrix<float, 8, 8> C", "simdgroup_matrix<half, 8, 8> C")
        source = source.replace(
            "C[ct] = simdgroup_matrix<float, 8, 8>(0)", "C[ct] = simdgroup_matrix<half, 8, 8>(0)"
        )
        store = "for (int ct = 0; ct < TC * RT; ++ct) simdgroup_store(C[ct], red + (sg * TC * RT + ct) * 64, 8);"
        replacement = """
        for (int ct = 0; ct < TC * RT; ++ct) {
            simdgroup_matrix<float, 8, 8> cf;
            cf.thread_elements()[0] = float(C[ct].thread_elements()[0]);
            cf.thread_elements()[1] = float(C[ct].thread_elements()[1]);
            simdgroup_store(cf, red + (sg * TC * RT + ct) * 64, 8);
        }
        """
        if accumulator == "float16_chunked":
            source = source.replace(
                "int kbeg = (int)sg * KPS;",
                """
                simdgroup_matrix<float, 8, 8> totals[TC * RT];
                for (int ct = 0; ct < TC * RT; ++ct)
                    totals[ct] = simdgroup_matrix<float, 8, 8>(0);
                int kbeg = (int)sg * KPS;
            """,
            )
            boundary = "        simdgroup_barrier(mem_flags::mem_threadgroup);\n    }"
            source = source.replace(
                boundary,
                """
                for (int ct = 0; ct < TC * RT; ++ct) {
                    totals[ct].thread_elements()[0] += float(C[ct].thread_elements()[0]);
                    totals[ct].thread_elements()[1] += float(C[ct].thread_elements()[1]);
                    C[ct] = simdgroup_matrix<half, 8, 8>(0);
                }
            """
                + boundary,
            )
            replacement = store.replace("C[ct]", "totals[ct]")
        source = source.replace(store, replacement)
    return mx.fast.metal_kernel(
        name=f"geist_lab_{bits}_{math_mode}_{accumulator}",
        input_names=inputs,
        output_names=["out"],
        source=source,
        compile_options={"math_mode": math_mode},
    )


def lab_matmul(linear, x, variant: str, *, split_k: int = 8, pad_rows: int = 16):
    rows, k = math.prod(x.shape[:-1]), x.shape[-1]
    n = linear.weight.shape[0]
    flat = x.reshape(rows, k)
    if variant in {"pad", "fp16_native"}:
        padded = max(rows, pad_rows) if variant == "pad" else rows
        if padded != rows:
            flat = mx.concatenate([flat, mx.zeros((padded - rows, k), dtype=flat.dtype)])
        if variant == "fp16_native":
            # Explicit precision variant: Q4 integers stay unchanged, but both
            # activation and dequantization rounding differ from BF16.
            out = mx.quantized_matmul(
                flat.astype(mx.float16),
                linear.weight,
                linear.scales.astype(mx.float16),
                linear.biases.astype(mx.float16),
                transpose=True,
                group_size=64,
                bits=linear.bits,
            ).astype(x.dtype)
        else:
            out = mx.quantized_matmul(
                flat,
                linear.weight,
                linear.scales,
                linear.biases,
                transpose=True,
                group_size=64,
                bits=linear.bits,
            )
        out = out[:rows].reshape(*x.shape[:-1], n)
        return out + linear.bias if "bias" in linear else out
    if variant == "staged":
        return small_m_matmul(linear, x, split_k)
    if variant == "native":
        return linear(x)
    if variant not in {"relaxed", "fast", "half_acc", "chunked_half_acc", "affine_fold"}:
        raise ValueError(f"Unknown experimental projection {variant}")
    accumulator = {
        "half_acc": "float16",
        "chunked_half_acc": "float16_chunked",
        "affine_fold": "affine_fold",
    }.get(variant, "float32")
    math_mode = variant if variant in {"relaxed", "fast"} else "safe"
    kernel = experimental_kernel(linear.bits, math_mode, accumulator)
    if variant == "affine_fold" and (linear.bits != 4 or n % 8):
        return linear(x)
    row_tiles = (rows + 7) // 8
    if rows > 32 or k % (split_k * 64):
        raise ValueError("Experimental MMA requires up to 32 rows and aligned split-K")
    if rows != row_tiles * 8:
        flat = mx.concatenate([flat, mx.zeros((row_tiles * 8 - rows, k), dtype=x.dtype)])
    operand_dtype = mx.float16 if accumulator.startswith("float16") else mx.bfloat16
    (out,) = kernel(
        inputs=[flat.astype(operand_dtype), linear.weight, linear.scales, linear.biases]
        + (
            [flat.astype(mx.float32).reshape(row_tiles * 8, k // 64, 64).sum(-1)]
            if variant == "affine_fold"
            else []
        ),
        template=[
            ("KD", k),
            ("ND", n),
            ("MD", rows),
            ("SK", split_k),
            ("TC", 1),
            ("RT", row_tiles),
            ("T", operand_dtype),
            ("PACKED", False),
            ("GB", 1),
            ("BN", 8),
        ],
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
        grid=(((n + 7) // 8) * split_k * 32, 1, 1),
        threadgroup=(split_k * 32, 1, 1),
    )
    out = out.reshape(*x.shape[:-1], n)
    return out + linear.bias if "bias" in linear else out


class LabLinear(nn.Module):
    def __init__(self, linear, variant: str, pad_rows: int):
        super().__init__()
        self.linear, self.variant, self.pad_rows = linear, variant, pad_rows
        self.enabled = True

    def __call__(self, x):
        rows = math.prod(x.shape[:-1])
        if self.enabled and 6 <= rows <= 32 and x.dtype == mx.bfloat16:
            return lab_matmul(self.linear, x, self.variant, pad_rows=self.pad_rows)
        return self.linear(x)


def install_lab(model, variant: str, pad_rows: int = 16):
    replacements = []
    wrapped = []
    for name, layer in model.named_modules():
        if isinstance(layer, LabLinear):
            wrapped.append(name + ".")
            continue
        if any(name.startswith(prefix) for prefix in wrapped):
            continue
        if not isinstance(layer, nn.QuantizedLinear):
            continue
        if layer.mode != "affine" or layer.bits not in (4, 8) or layer.group_size != 64:
            continue
        k = layer.weight.shape[1] * (32 // layer.bits)
        if k % 512 or layer.weight.shape[0] < 4096:
            continue
        replacements.append((name, LabLinear(layer, variant, pad_rows)))
    model.update_modules(tree_unflatten(replacements))
    return [layer for _, layer in replacements]
