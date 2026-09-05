"""Qwen hybrid target with captured recurrent inputs for speculative rollback.

Forward math follows mlx-lm 0.31.3 (Apple, MIT); see THIRD_PARTY_NOTICES.md.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.gated_delta import gated_delta_update


@dataclass
class DeltaInputs:
    q: mx.array
    k: mx.array
    v: mx.array
    a: mx.array
    b: mx.array
    state: mx.array | None
    conv_input: mx.array
    mask: mx.array | None


def captured_delta(layer, inputs, mask, cache):
    batch, width, _ = inputs.shape
    if layer.sharding_group is not None or cache.lengths is not None:
        raise ValueError("Speculative Qwen requires an unsharded single sequence")
    qkv = layer.in_proj_qkv(inputs)
    z = layer.in_proj_z(inputs).reshape(batch, width, layer.num_v_heads, layer.head_v_dim)
    b = layer.in_proj_b(inputs)
    a = layer.in_proj_a(inputs)
    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros(
            (batch, layer.conv_kernel_size - 1, layer.conv_dim), dtype=inputs.dtype
        )
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    cache[0] = mx.contiguous(conv_input[:, -(layer.conv_kernel_size - 1) :, :])
    conv_out = nn.silu(layer.conv1d(conv_input))
    q, k, v = [
        t.reshape(batch, width, heads, dim)
        for t, heads, dim in zip(
            mx.split(conv_out, [layer.key_dim, 2 * layer.key_dim], -1),
            [layer.num_k_heads, layer.num_k_heads, layer.num_v_heads],
            [layer.head_k_dim, layer.head_k_dim, layer.head_v_dim],
            strict=False,
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    record = DeltaInputs(q, k, v, a, b, cache[1], conv_input, mask)
    out, cache[1] = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        cache[1],
        mask,
        use_kernel=not layer.training,
    )
    cache.advance(width)
    out = layer.norm(out, z)
    return layer.out_proj(out.reshape(batch, width, -1)), record


class QwenSpeculativeTarget:
    def __init__(self, model, tap_layers: tuple[int, ...]):
        if model.model_type not in {"qwen3_5", "qwen3_5_text"}:
            raise ValueError("DFlash backend currently supports dense Qwen 3.5/3.8 only")
        self.model = model
        self.language_model = getattr(model, "language_model", model)
        self.body = self.language_model.model
        if self.language_model.args.num_experts:
            raise ValueError("DFlash backend has not been qualified for MoE targets")
        if tuple(sorted(set(tap_layers))) != tap_layers:
            raise ValueError("Drafter tap layers must be unique and sorted")
        if not tap_layers or min(tap_layers) < 0 or max(tap_layers) >= len(self.body.layers):
            raise ValueError("Drafter tap layers are outside the target")
        self.tap_layers = frozenset(tap_layers)
        self.records = {}
        self.verify_width = 0

    def make_cache(self):
        return make_prompt_cache(self.model)

    def forward(self, ids, cache, *, capture: bool = False, last_only: bool = False):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Speculative Qwen requires batch size one")
        h = self.body.embed_tokens(ids)
        fa_mask = create_attention_mask(h, cache[self.body.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.body.ssm_idx])
        tapped = []
        if capture:
            self.records = {}
            self.verify_width = ids.shape[1]
        for index, (layer, state) in enumerate(zip(self.body.layers, cache, strict=False)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if capture and layer.is_linear:
                r, record = captured_delta(layer.linear_attn, layer.input_layernorm(h), mask, state)
                self.records[index] = record
                h = h + r
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            else:
                h = layer(h, mask=mask, cache=state)
            if index in self.tap_layers:
                tapped.append(h)
        h = self.body.norm(h[:, -1:] if last_only else h)
        head = getattr(self.language_model, "lm_head", self.body.embed_tokens.as_linear)
        return head(h), mx.concatenate(tapped, axis=-1)

    def rollback(self, cache, keep: int) -> None:
        if not 1 <= keep <= self.verify_width:
            raise ValueError("Rollback must retain the anchor and at most the verify width")
        rejected = self.verify_width - keep
        if rejected:
            for index, (layer, state) in enumerate(zip(self.body.layers, cache, strict=False)):
                if not layer.is_linear:
                    state.trim(rejected)
                    continue
                r = self.records[index]
                gdn = layer.linear_attn
                mask = r.mask[:, :keep] if r.mask is not None else None
                _, state[1] = gated_delta_update(
                    r.q[:, :keep],
                    r.k[:, :keep],
                    r.v[:, :keep],
                    r.a[:, :keep],
                    r.b[:, :keep],
                    gdn.A_log,
                    gdn.dt_bias,
                    r.state,
                    mask,
                    use_kernel=not gdn.training,
                )
                state[0] = mx.contiguous(r.conv_input[:, keep : keep + gdn.conv_kernel_size - 1])
        self.records = {}
