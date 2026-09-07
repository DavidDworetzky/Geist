"""Experimental compiled recurrent layers with explicit, rollback-safe state."""

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache

from agents.architectures.llama.qwen_speculative import (
    DeltaInputs,
    QwenSpeculativeTarget,
    captured_delta,
)


def compile_recurrent_layer(layer):
    def step(h, conv, recurrent, mask):
        cache = ArraysCache(2)
        cache[0], cache[1] = conv, recurrent
        r, record = captured_delta(layer.linear_attn, layer.input_layernorm(h), mask, cache)
        h = h + r
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return (
            h,
            cache[0],
            cache[1],
            record.q,
            record.k,
            record.v,
            record.a,
            record.b,
            record.conv_input,
        )

    return mx.compile(step)


class CompiledQwenTarget(QwenSpeculativeTarget):
    def __init__(self, model, tap_layers: tuple[int, ...]):
        super().__init__(model, tap_layers)
        self.steps = {
            i: compile_recurrent_layer(layer)
            for i, layer in enumerate(self.body.layers)
            if layer.is_linear
        }

    def forward(self, ids, cache, *, capture: bool = False, last_only: bool = False):
        if not capture:
            return super().forward(ids, cache, capture=capture, last_only=last_only)
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Compiled verifier requires batch size one")
        h = self.body.embed_tokens(ids)
        fa_mask = create_attention_mask(h, cache[self.body.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.body.ssm_idx])
        self.records = {}
        self.verify_width = ids.shape[1]
        tapped = []
        for i, (layer, state) in enumerate(zip(self.body.layers, cache, strict=True)):
            if layer.is_linear:
                if state.lengths is not None or layer.linear_attn.sharding_group is not None:
                    raise ValueError("Compiled verifier requires unsharded, single-sequence state")
                previous = state[1]
                h, state[0], state[1], q, k, v, a, b, conv = self.steps[i](
                    h, state[0], previous, ssm_mask
                )
                state.advance(ids.shape[1])
                self.records[i] = DeltaInputs(q, k, v, a, b, previous, conv, ssm_mask)
            else:
                h = layer(h, mask=fa_mask, cache=state)
            if i in self.tap_layers:
                tapped.append(h)
        h = self.body.norm(h[:, -1:] if last_only else h)
        head = getattr(self.language_model, "lm_head", self.body.embed_tokens.as_linear)
        return head(h), mx.concatenate(tapped, axis=-1)
