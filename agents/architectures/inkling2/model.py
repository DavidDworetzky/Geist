"""Top-level Inkling 2 / Inkling 2 Mini text-generation MLX model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

from agents.architectures.inkling2 import checkpoint
from agents.architectures.inkling2.config import ModelArgs, TextArgs
from agents.architectures.inkling2.layers import (
    BlockAttentionResidual,
    DecoderLayer,
    FullDecoderLayer,
    MessageInput,
    SlidingDecoderLayer,
    unweighted_rms_norm,
)


__all__ = ["Inkling2Model", "Model", "TextModel"]


class TextModel(nn.Module):
    def __init__(self, args: TextArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers: list[DecoderLayer] = [
            (
                SlidingDecoderLayer(args, i)
                if i in args.local_layer_ids
                else FullDecoderLayer(args, i)
            )
            for i in range(args.num_hidden_layers)
        ]
        self.final_input = BlockAttentionResidual(args.hidden_size)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if args.loop_config is not None:
            self.message_input = MessageInput(args.hidden_size, len(args.loop_core))

    def embed(self, inputs: mx.array) -> mx.array:
        return self.embed_norm(self.embed_tokens(inputs))

    @property
    def layer_visits(self) -> list[DecoderLayer]:
        if self.args.loop_config is None:
            return self.layers
        core = self.args.loop_core
        passes = int(self.args.loop_config["num_passes"])
        return (
            self.layers[: core.start]
            + self.layers[core.start : core.stop] * passes
            + self.layers[core.stop :]
        )

    def _run_loop(
        self,
        hidden_states: mx.array,
        cache: list[Any],
        conv_mask: mx.array | None,
        tap_indices: set[int] | None = None,
    ) -> tuple[mx.array, mx.array, dict[int, mx.array]]:
        core = self.args.loop_core
        caches = iter(cache)
        taps: dict[int, mx.array] = {}
        visit_idx = 0

        def run(
            layers: list[DecoderLayer],
            states: mx.array,
            block_states: mx.array | None,
            bank: tuple[mx.array, mx.array] | None = None,
        ) -> tuple[mx.array, mx.array]:
            nonlocal visit_idx
            for i, layer in enumerate(layers):
                if tap_indices is not None and visit_idx in tap_indices:
                    taps[visit_idx] = states
                messages = None if bank is None else self.message_input(bank, i)
                states, block_states = layer(
                    states,
                    block_states,
                    cache=next(caches),
                    conv_mask=conv_mask,
                    messages=messages,
                )
                visit_idx += 1
            return states, block_states

        prelude = run(self.layers[: core.start], hidden_states, None)
        hidden_states, block_states = prelude
        assert self.args.loop_config is not None
        for loop_pass in range(int(self.args.loop_config["num_passes"])):
            bank = None
            if loop_pass > 0:
                bank = (
                    block_states[prelude[1].shape[0] + 1 :],
                    unweighted_rms_norm(hidden_states.astype(mx.float32)),
                )
                hidden_states, block_states = prelude
            hidden_states, block_states = run(
                self.layers[core.start : core.stop], hidden_states, block_states, bank
            )
        hidden_states, block_states = run(self.layers[core.stop :], hidden_states, block_states)
        return hidden_states, block_states, taps

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        return_taps: set[int] | None = None,
    ) -> mx.array | tuple[mx.array, dict[int, mx.array]]:
        if input_embeddings is None:
            input_embeddings = self.embed(inputs)
        hidden_states = input_embeddings
        dtype = hidden_states.dtype

        if cache is None:
            cache = [None] * len(self.layer_visits)
        conv_mask = None
        if cache[0] is not None:
            conv_mask = cache[0][1].make_mask(hidden_states.shape[1])

        taps: dict[int, mx.array] = {}
        block_states = None
        if self.args.loop_config is None:
            for visit_idx, (layer, c) in enumerate(zip(self.layers, cache, strict=False)):
                if return_taps is not None and visit_idx in return_taps:
                    taps[visit_idx] = hidden_states
                hidden_states, block_states = layer(
                    hidden_states, block_states, cache=c, conv_mask=conv_mask
                )
        else:
            hidden_states, block_states, taps = self._run_loop(
                hidden_states, cache, conv_mask, tap_indices=return_taps
            )

        normed_states = unweighted_rms_norm(hidden_states.astype(mx.float32))
        hidden_states = self.final_input(block_states, normed_states)[..., 0, :]
        hidden_states = self.norm(hidden_states).astype(dtype)
        if return_taps is not None:
            final_idx = len(self.layer_visits)
            if final_idx in return_taps:
                taps[final_idx] = hidden_states
            return hidden_states, taps
        return hidden_states


class Inkling2Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.language_model = TextModel(args.text)

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.language_model.layers


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Inkling2Model(args)
        text = args.text
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.model.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings
        )
        logits = self.lm_head(hidden_states)
        text = self.args.text
        softcap = text.final_logit_softcapping
        logits = logits.astype(mx.float32)
        logits = mx.tanh(logits / softcap) * softcap
        return logits[..., : text.unpadded_vocab_size]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return checkpoint.sanitize(self, weights)

    @property
    def quant_predicate(self) -> Callable[[str, nn.Module], bool]:
        def predicate(path: str, module: nn.Module) -> bool:
            return not path.startswith("model.vision_tower.") and ("_message_gate." not in path)

        return predicate

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.language_model.layers

    def make_cache(self) -> list[CacheList]:
        window = self.args.text.sliding_window_size
        return [
            CacheList(
                (
                    RotatingKVCache(max_size=window)
                    if isinstance(layer, SlidingDecoderLayer)
                    else KVCache()
                ),
                ArraysCache(4),
            )
            for layer in self.model.language_model.layer_visits
        ]
