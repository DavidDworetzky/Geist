"""Small-row quantized Metal matmul for speculative verification.

Automatic routing uses the staged eight-row layout. Wider tiles, packing,
register fragments, lookup tables, and padded stages are benchmark experiments.

Kernel adapted from avlp12/mlx-lm fast_qmm.py via ARahim3/mlx-dspark
d2719285af03e875efbb540c459c5971947ac015. See THIRD_PARTY_NOTICES.md.
"""

import math
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten


_UNPACK = {
    4: r"""
            const device uint* wr = PACKED
                ? w + (((size_t)(n / 8) * (K / 64) + g) * 32 + lane) * 2
                : w + (size_t)n * (K / 8) + (ka >> 3) + kq * 2;
            uint p0 = wr[0], p1 = wr[1];
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + t) * BN + j] = (T)(bfloat16_t)((float)((p0 >> (4 * t)) & 15u) * s + bb);
            for (int t = 0; t < 8; ++t)
                bt[(kq * 16 + 8 + t) * BN + j] = (T)(bfloat16_t)((float)((p1 >> (4 * t)) & 15u) * s + bb);
""",
    8: r"""
            const device uint* wr = PACKED
                ? w + (((size_t)(n / 8) * (K / 64) + g) * 32 + lane) * 4
                : w + (size_t)n * (K / 4) + (ka >> 2) + kq * 4;
            for (int u = 0; u < 4; ++u) {
                uint p = wr[u];
                for (int t = 0; t < 4; ++t)
                    bt[(kq * 16 + u * 4 + t) * BN + j] =
                        (T)(bfloat16_t)((float)((p >> (8 * t)) & 255u) * s + bb);
            }
""",
}

_SRC = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / SK;                // K-span per simdgroup (split-K)

    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;

    int n0 = (int)tgid * 8 * TC;

    // x is read straight from device into the MMA (bf16 in, fp32 accumulate): no
    // staging keeps threadgroup memory small and, more importantly, the critical
    // path short.
    threadgroup T bs[SK * TC * GB * 64 * BN];
    threadgroup float red[SK * TC * RT * 64];

    simdgroup_matrix<float, 8, 8> C[TC * RT];
    for (int ct = 0; ct < TC * RT; ++ct) C[ct] = simdgroup_matrix<float, 8, 8>(0);

    // Split-K: the 8 simdgroups each walk 1/8 of K in short serial loops. (A prior
    // revision walked all of K per threadgroup and put ~160 barrier pairs on the
    // critical path.)
    int kbeg = (int)sg * KPS;
    for (int kk = 0; kk < KPS; kk += GB * 64) {
        int j  = (int)(lane & 7);
        int kq = (int)(lane >> 3);
        for (int ct = 0; ct < TC; ++ct) {
        for (int cg = 0; cg < GB; ++cg) {
        int ka = kbeg + kk + cg * 64;
        threadgroup T* bt = bs + ((sg * TC + ct) * GB + cg) * 64 * BN;
        int n  = n0 + ct * 8 + j;
        if (n < N) {
            // dequantize per quantization group (64 values), not per MMA tile (8):
            // one scale/bias load serves the whole group and barriers drop 8x.
            int g = ka >> 6;
            size_t qi = PACKED ? ((size_t)(n / 8) * (K / 64) + g) * 8 + (n & 7)
                               : (size_t)n * (K / 64) + g;
            float s  = (float)sc[qi];
            float bb = (float)bi[qi];
__UNPACK__
        } else {
            for (int t = 0; t < 16; ++t) bt[(kq * 16 + t) * BN + j] = (T)0;
        }
        }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<T, 8, 8> A, B;
        for (int kt = 0; kt < GB * 8; ++kt) {
            for (int rt = 0; rt < RT; ++rt) {
            simdgroup_load(A, x + rt * 8 * K + kbeg + kk + kt * 8, K);
            for (int ct = 0; ct < TC; ++ct) {
                simdgroup_load(B, bs + (sg * TC + ct) * GB * 64 * BN + kt * 8 * BN, BN);
                simdgroup_multiply_accumulate(C[rt * TC + ct], A, B, C[rt * TC + ct]);
            }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int ct = 0; ct < TC * RT; ++ct) simdgroup_store(C[ct], red + (sg * TC * RT + ct) * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sum the 8 split-K partials and write out
    for (int i = (int)tid; i < TC * RT * 64; i += SK * 32) {
        int m = (i / (TC * 64)) * 8 + ((i >> 3) & 7), j = i & 7;
        int n = n0 + ((i >> 6) % TC) * 8 + j;
        if (m < M && n < N) {
            float v = 0.0f;
            for (int q = 0; q < SK; ++q) v += red[q * TC * RT * 64 + i];
            out[(size_t)m * N + n] = (bfloat16_t)v;
        }
    }
"""

_KERNELS = {
    bits: mx.fast.metal_kernel(
        name=f"geist_qmm_mma{bits}",
        input_names=["x", "w", "sc", "bi"],
        output_names=["out"],
        source=_SRC.replace("__UNPACK__", unpack),
    )
    for bits, unpack in _UNPACK.items()
}

_LUT_UNPACK = _UNPACK[4]
for _word in ("p0", "p1"):
    _LUT_UNPACK = _LUT_UNPACK.replace(
        f"(T)(bfloat16_t)((float)(({_word} >> (4 * t)) & 15u) * s + bb)",
        f"(T)table[qi * 16 + (({_word} >> (4 * t)) & 15u)]",
    )
_LUT_KERNEL = mx.fast.metal_kernel(
    name="geist_qmm_lookup",
    input_names=["x", "w", "sc", "bi", "table"],
    output_names=["out"],
    source=_SRC.replace("__UNPACK__", _LUT_UNPACK),
)


# Fragment coordinates follow MLX Steel's BaseMMAFrag::get_coord (Apple, MIT).
_DIRECT_SOURCE = r"""
    const int K = KD, N = ND, M = MD;
    const int KPS = KD / SK, PACK = 32 / BITS;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid >> 5, lane = tid & 31;
    int n0 = (int)threadgroup_position_in_grid.x * 8;
    int qid = lane / 4;
    int fm = (qid & 4) + ((lane / 2) % 4);
    int fn = (qid & 2) * 2 + (lane % 2) * 2;
    int n1 = n0 + fn, n2 = n1 + 1;
    threadgroup float red[SK * RT * 64];
    simdgroup_matrix<float, 8, 8> C[RT];
    for (int rt = 0; rt < RT; ++rt) C[rt] = simdgroup_matrix<float, 8, 8>(0);
    for (int kk = 0; kk < KPS; kk += 64) {
        int ka = (int)sg * KPS + kk, g = ka / 64;
        size_t si1 = PACKED ? ((size_t)(n1 / 8) * (K / 64) + g) * 8 + (n1 & 7)
                            : (size_t)n1 * (K / 64) + g;
        size_t si2 = PACKED ? si1 + 1 : si1 + K / 64;
        float s1 = n1 < N ? (float)sc[si1] : 0.0f;
        float s2 = n2 < N ? (float)sc[si2] : 0.0f;
        float b1 = n1 < N ? (float)bi[si1] : 0.0f;
        float b2 = n2 < N ? (float)bi[si2] : 0.0f;
        size_t pre1 = PACKED
            ? ((((size_t)(n1 / 8) * (K / 64) + g) * 4 + fm / 2) * 8 + (n1 & 7)) * (BITS / 2) + (fm % 2) * (BITS / 4)
            : (size_t)n1 * (K / PACK) + ka / PACK + fm * (BITS / 4);
        size_t pre2 = PACKED ? pre1 + BITS / 2 : pre1 + K / PACK;
        uint a1 = SHUFFLE && n1 < N ? w[pre1] : 0;
        uint a2 = SHUFFLE && n2 < N ? w[pre2] : 0;
        uint c1 = SHUFFLE && BITS == 8 && n1 < N ? w[pre1 + 1] : 0;
        uint c2 = SHUFFLE && BITS == 8 && n2 < N ? w[pre2 + 1] : 0;
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            int ik = ka + kt * 8 + fm;
            size_t wi1 = PACKED
                ? ((((size_t)(n1 / 8) * (K / 64) + g) * 4 + (ik % 64) / 16) * 8 + (n1 & 7)) * (BITS / 2) + (ik % 16) / PACK
                : (size_t)n1 * (K / PACK) + ik / PACK;
            size_t wi2 = PACKED ? wi1 + BITS / 2 : wi1 + K / PACK;
            uint p1, p2;
            if (SHUFFLE) {
                uint source_lane = (lane & 9) | ((kt & 4) << 2) | ((kt & 3) << 1);
                uint lo1 = simd_shuffle(a1, source_lane), lo2 = simd_shuffle(a2, source_lane);
                uint hi1 = simd_shuffle(c1, source_lane), hi2 = simd_shuffle(c2, source_lane);
                p1 = BITS == 8 && fm >= 4 ? hi1 : lo1;
                p2 = BITS == 8 && fm >= 4 ? hi2 : lo2;
            } else {
                p1 = n1 < N ? w[wi1] : 0;
                p2 = n2 < N ? w[wi2] : 0;
            }
            uint shift = (ik % PACK) * BITS;
            simdgroup_matrix<T, 8, 8> A, B;
            B.thread_elements()[0] = (T)(bfloat16_t)((float)((p1 >> shift) & ((1u << BITS) - 1)) * s1 + b1);
            B.thread_elements()[1] = (T)(bfloat16_t)((float)((p2 >> shift) & ((1u << BITS) - 1)) * s2 + b2);
            for (int rt = 0; rt < RT; ++rt) {
                simdgroup_load(A, x + rt * 8 * K + ka + kt * 8, K);
                simdgroup_multiply_accumulate(C[rt], A, B, C[rt]);
            }
        }
    }
    for (int rt = 0; rt < RT; ++rt) simdgroup_store(C[rt], red + (sg * RT + rt) * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int)tid; i < RT * 64; i += SK * 32) {
        int m = i / 8, n = n0 + (i & 7);
        if (m < M && n < N) {
            float value = 0.0f;
            for (int part = 0; part < SK; ++part) value += red[part * RT * 64 + i];
            out[(size_t)m * N + n] = (bfloat16_t)value;
        }
    }
"""

_DIRECT_KERNEL = mx.fast.metal_kernel(
    name="geist_qmm_direct",
    input_names=["x", "w", "sc", "bi"],
    output_names=["out"],
    source=_DIRECT_SOURCE,
)


def small_m_matmul(
    linear,
    x,
    split_k: int = 8,
    *,
    half_operands: bool = False,
    column_tiles: int = 1,
    packed_weights=None,
    direct_fragment: bool = False,
    shuffle_fragments: bool = False,
    group_batch: int = 1,
    stage_stride: int = 8,
    dequant_table=None,
):
    rows = math.prod(x.shape[:-1])
    if not 1 <= rows <= 32:
        raise ValueError("The tiled kernel supports 1..32 rows")
    row_tiles = (rows + 7) // 8
    k = x.shape[-1]
    n = linear.weight.shape[0]
    if not 1 <= split_k <= 32 or k % (split_k * 64):
        raise ValueError("split_k must divide the input's quantization groups")
    if group_batch not in (1, 2, 4) or k % (split_k * group_batch * 64):
        raise ValueError("group_batch must divide each split's quantization groups")
    if stage_stride not in (8, 9, 10, 12, 16):
        raise ValueError("Unsupported staging row stride")
    if (
        column_tiles not in (1, 2, 4)
        or split_k * column_tiles * (group_batch * 128 * stage_stride + row_tiles * 256) > 32768
    ):
        raise ValueError("Column tiles exceed the kernel's threadgroup memory budget")
    if direct_fragment and column_tiles != 1:
        raise ValueError("Direct fragments currently use one column tile")
    if dequant_table is not None and (
        linear.bits != 4 or packed_weights is not None or direct_fragment
    ):
        raise ValueError("Lookup dequantization requires unpacked 4-bit staged weights")
    threadgroup = split_k * 32
    flat = x.reshape(rows, k)
    operand_dtype = mx.float16 if half_operands else mx.bfloat16
    if rows < row_tiles * 8:
        flat = mx.concatenate([flat, mx.zeros((row_tiles * 8 - rows, k), dtype=x.dtype)])
    flat = flat.astype(operand_dtype)
    kernel = _DIRECT_KERNEL if direct_fragment else _KERNELS[linear.bits]
    if dequant_table is not None:
        kernel = _LUT_KERNEL
    (out,) = kernel(
        inputs=[
            flat,
            *(
                packed_weights
                if packed_weights is not None
                else (linear.weight, linear.scales, linear.biases)
            ),
            *([dequant_table] if dequant_table is not None else []),
        ],
        template=[
            ("KD", k),
            ("ND", n),
            ("MD", rows),
            ("SK", split_k),
            ("TC", column_tiles),
            ("RT", row_tiles),
            ("T", operand_dtype),
            ("PACKED", packed_weights is not None),
            ("BITS", linear.bits),
            ("SHUFFLE", shuffle_fragments),
            ("GB", group_batch),
            ("BN", stage_stride),
        ],
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
        grid=(((n + 8 * column_tiles - 1) // (8 * column_tiles)) * threadgroup, 1, 1),
        threadgroup=(threadgroup, 1, 1),
    )
    out = out.reshape(*x.shape[:-1], n)
    return out + linear.bias if "bias" in linear else out


def pack_small_m_weights(linear) -> tuple[mx.array, mx.array, mx.array]:
    """Reorder compressed groups so each SIMD group's weight loads are contiguous."""
    n = linear.weight.shape[0]
    k = linear.weight.shape[1] * (32 // linear.bits)
    if n % 8 or k % 64 or linear.bits not in (4, 8):
        raise ValueError("Packing requires 8-column tiles and 64-element affine groups")
    weight = linear.weight.reshape(n // 8, 8, k // 64, 4, linear.bits // 2)
    weight = mx.contiguous(weight.transpose(0, 2, 3, 1, 4)).reshape(-1)
    scales = mx.contiguous(linear.scales.reshape(n // 8, 8, k // 64).transpose(0, 2, 1)).reshape(-1)
    biases = mx.contiguous(linear.biases.reshape(n // 8, 8, k // 64).transpose(0, 2, 1)).reshape(-1)
    return weight, scales, biases


def make_dequant_table(linear) -> mx.array:
    if linear.bits != 4 or linear.group_size != 64:
        raise ValueError("Lookup dequantization requires 4-bit affine groups of 64")
    values = mx.arange(16, dtype=mx.float32)
    table = (
        linear.scales.astype(mx.float32)[..., None] * values
        + linear.biases.astype(mx.float32)[..., None]
    )
    return table.astype(mx.bfloat16).reshape(-1)


class SmallMLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.enabled = True
        self.qualified = True
        self.split_k = 8
        self.half_operands = False
        self.column_tiles = 1
        self.packed_weights = None
        self.max_rows = 8
        self.direct_fragment = False
        self.shuffle_fragments = False
        self.group_batch = 1
        self.stage_stride = 8

    def __call__(self, x):
        if (
            self.enabled
            and self.qualified
            and x.dtype == mx.bfloat16
            and 6 <= math.prod(x.shape[:-1]) <= self.max_rows
        ):
            return small_m_matmul(
                self.linear,
                x,
                self.split_k,
                half_operands=self.half_operands,
                column_tiles=self.column_tiles,
                packed_weights=self.packed_weights,
                direct_fragment=self.direct_fragment,
                shuffle_fragments=self.shuffle_fragments,
                group_batch=self.group_batch,
                stage_stride=self.stage_stride,
            )
        return self.linear(x)


def install_small_m(model) -> list[SmallMLinear]:
    replacements = []
    modules = list(model.named_modules())
    wrapped = [name for name, layer in modules if isinstance(layer, SmallMLinear)]
    for name, layer in modules:
        if any(name.startswith(prefix + ".") for prefix in wrapped):
            continue
        if not isinstance(layer, nn.QuantizedLinear):
            continue
        if layer.mode != "affine" or layer.bits not in (4, 8) or layer.group_size != 64:
            continue
        k = layer.weight.shape[1] * (32 // layer.bits)
        if "biases" not in layer or k % 512 or layer.weight.shape[0] < 4096:
            continue
        if layer.scales.dtype != mx.bfloat16 or layer.biases.dtype != mx.bfloat16:
            continue
        replacements.append((name, SmallMLinear(layer)))
    model.update_modules(tree_unflatten(replacements))
    return [layer for _, layer in replacements]


def tune_small_m(wrappers: list[SmallMLinear]) -> list[dict]:
    """Qualify each weight shape against native QMM on this GPU, once per load."""
    groups = {}
    for wrapper in wrappers:
        layer = wrapper.linear
        k = layer.weight.shape[1] * (32 // layer.bits)
        groups.setdefault((k, layer.weight.shape[0], layer.bits), []).append(wrapper)
    results = []
    for (k, n, bits), group in groups.items():
        # Rotate real weights to avoid timing only a repeatedly cached projection.
        layers = [wrapper.linear for wrapper in group[:3]]
        inputs = [
            mx.random.normal((1, 8, k), key=mx.random.key(11 + index)).astype(mx.bfloat16)
            for index in range(len(layers))
        ]
        mx.eval(inputs)
        reference = [layer(x) for layer, x in zip(layers, inputs, strict=True)]
        mx.eval(reference)
        timings = {}
        for split_k in (0, 2, 4, 8):

            def project(index: int, x: mx.array, group=group, split_k=split_k) -> mx.array:
                wrapper = group[index]
                return (
                    small_m_matmul(
                        wrapper.linear,
                        x,
                        split_k,
                        packed_weights=wrapper.packed_weights,
                        half_operands=wrapper.half_operands,
                        column_tiles=wrapper.column_tiles,
                        direct_fragment=wrapper.direct_fragment,
                        shuffle_fragments=wrapper.shuffle_fragments,
                        group_batch=wrapper.group_batch,
                        stage_stride=wrapper.stage_stride,
                    )
                    if split_k
                    else wrapper.linear(x)
                )

            try:
                outputs = [project(index, x) for index, x in enumerate(inputs)]
            except ValueError:
                continue
            mx.eval(outputs)
            error = (
                mx.stack(
                    [
                        mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))
                        / mx.maximum(mx.max(mx.abs(ref.astype(mx.float32))), 1e-9)
                        for out, ref in zip(outputs, reference, strict=True)
                    ]
                )
                .max()
                .item()
            )
            if error > 0.02:
                continue
            samples = []
            for trial in range(4):
                started = time.perf_counter()
                output = project(0, inputs[0])
                for step in range(1, 12):
                    index = step % len(layers)
                    # Preserve a real data dependency: independent projection
                    # launches can overlap, but transformer layers cannot.
                    x = inputs[index] + mx.tanh(output[..., :1]) * 1e-4
                    output = project(index, x)
                mx.eval(output)
                if trial:
                    samples.append(time.perf_counter() - started)
            timings[split_k] = statistics.median(samples)
        winner = min(timings, key=timings.get)
        qualified = winner != 0 and timings[winner] < timings[0] / 1.08
        for wrapper in group:
            wrapper.qualified = qualified
            wrapper.split_k = winner if qualified else 8
        results.append(
            {
                "k": k,
                "n": n,
                "bits": bits,
                "split_k": winner if qualified else 0,
                "speedup": timings[0] / timings[winner],
                "dependent_chain": True,
                "chain_length": 12,
            }
        )
    return results
