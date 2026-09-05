"""In-process DFlash 2 drafting and verification using Geist's loaded MLX target."""

import json
import time
from dataclasses import fields
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import wired_limit
from mlx_lm.sample_utils import apply_top_k, apply_top_p

from agents.architectures.llama.dflash_model import DFlashConfig, DFlashDraftModel
from agents.architectures.llama.qwen_speculative import QwenSpeculativeTarget


def load_drafter(path: str, target, bits: int = 8):
    root = Path(path).expanduser()
    raw = json.loads((root / "config.json").read_text())
    values = {**raw, **raw.get("dflash_config", {})}
    values["rope_theta"] = raw.get("rope_parameters", {}).get("rope_theta", 1000000.0)
    values["target_layer_ids"] = tuple(values["target_layer_ids"])
    allowed = {field.name for field in fields(DFlashConfig)}
    config = DFlashConfig(**{key: value for key, value in values.items() if key in allowed})
    if config.selector_rank <= 0 or config.block_size != 8:
        raise ValueError("Expected an eight-position DFlash 2 checkpoint")
    inner = getattr(target, "language_model", target)
    if config.hidden_size != inner.args.hidden_size or config.vocab_size != inner.args.vocab_size:
        raise ValueError("Drafter and target hidden/vocabulary dimensions do not match")
    if config.num_target_layers != inner.args.num_hidden_layers:
        raise ValueError("Drafter and target layer counts do not match")
    draft = DFlashDraftModel(config)
    weights = {}
    for shard in sorted(root.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    if not weights:
        raise ValueError(f"No safetensors drafter weights found in {root}")
    draft.load_weights(list(weights.items()), strict=True)
    if bits:
        nn.quantize(
            draft,
            group_size=64,
            bits=bits,
            class_predicate=lambda name, layer: isinstance(layer, nn.Linear)
            and "_conv" not in name
            and "candidate_selector" not in name,
        )
    draft.eval()
    mx.eval(draft.parameters())
    draft.bind(target)
    return draft


def target_probabilities(logits, temperature: float, top_p: float, top_k: int = 0):
    # Match MLX-LM: nucleus filtering is applied before temperature scaling.
    logprobs = logits.astype(mx.float32)
    logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
    if 0 < top_p < 1:
        logprobs = apply_top_p(logprobs, top_p)
    if 0 < top_k < logits.shape[-1]:
        logprobs = apply_top_k(logprobs, top_k)
    return mx.softmax(logprobs / temperature, axis=-1)


def sampled_accept(logits, proposals, candidate_ids, q_rows, temperature, top_p, top_k: int = 0):
    count = proposals.shape[0]
    p = target_probabilities(logits, temperature, top_p, top_k)
    q = mx.put_along_axis(mx.zeros_like(p[:count]), candidate_ids, q_rows, axis=1)
    indices = proposals[:, None]
    p_chosen = mx.take_along_axis(p[:count], indices, axis=1).squeeze(-1)
    q_chosen = mx.take_along_axis(q, indices, axis=1).squeeze(-1)
    matches = mx.random.uniform(shape=(count,)) < mx.minimum(
        1.0, p_chosen / mx.maximum(q_chosen, 1e-30)
    )
    accepted = int(mx.cumprod(matches.astype(mx.int32)).sum().item())
    distribution = p[-1] if accepted == count else mx.maximum(p[accepted] - q[accepted], 0)
    replacement = int(mx.random.categorical(mx.log(distribution)).item())
    return accepted, replacement


class DFlashDecoder:
    def __init__(
        self,
        model,
        tokenizer,
        drafter,
        *,
        max_draft: int = 7,
        prefill_step_size: int = 2048,
        profile: bool = False,
        experimental_block_size: int | None = None,
    ):
        self.block_size = (
            drafter.config.block_size
            if experimental_block_size is None
            else experimental_block_size
        )
        if not 8 <= self.block_size <= 32:
            raise ValueError("Experimental block size must be in 8..32")
        if not 1 <= max_draft < self.block_size:
            raise ValueError("DFlash max_draft must fit the draft block")
        if prefill_step_size <= 0:
            raise ValueError("Prefill step size must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.drafter = drafter
        self.target = QwenSpeculativeTarget(model, drafter.config.target_layer_ids)
        self.max_draft = max_draft
        self.prefill_step_size = prefill_step_size
        self.profile = profile
        self.last_stats = {}
        self._cached_tokens = ()
        self._cached_state = None

    def generate(
        self,
        prompt_tokens,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        if not prompt_tokens or max_tokens <= 0:
            raise ValueError("A nonempty prompt and positive output limit are required")
        if temperature < 0 or not 0 <= top_p <= 1 or top_k < 0:
            raise ValueError("Temperature and top_k must be nonnegative; top_p must be in [0, 1]")
        cached_count = len(self._cached_tokens)
        hit = (
            self._cached_state is not None
            and cached_count < len(prompt_tokens)
            and tuple(prompt_tokens[:cached_count]) == self._cached_tokens
        )
        saved = self._cached_state if hit else None
        self._cached_state = None
        self._cached_tokens = ()
        if saved is not None:
            cache, draft_cache, previous_context = saved
        else:
            cache = self.target.make_cache()
            draft_cache = self.drafter.make_cache()
            cached_count = 0
        stream = mx.new_thread_local_stream(mx.default_device())
        output = []
        acceptance = []
        phases = {"draft": 0.0, "verify": 0.0, "accept": 0.0, "rollback": 0.0}
        started = time.perf_counter()
        with wired_limit(self.model, [stream]), mx.stream(stream):
            if saved is not None:
                self.drafter.append_ctx(self.drafter.project_ctx(previous_context), draft_cache)
            context = None
            for start in range(cached_count, len(prompt_tokens), self.prefill_step_size):
                ids = mx.array([prompt_tokens[start : start + self.prefill_step_size]])
                logits, hidden = self.target.forward(ids, cache, last_only=True)
                if start + self.prefill_step_size < len(prompt_tokens):
                    self.drafter.append_ctx(self.drafter.project_ctx(hidden), draft_cache)
                    mx.eval([item.state for item in cache], [item.state for item in draft_cache])
                else:
                    context = hidden
            if temperature == 0:
                pending = int(mx.argmax(logits[0, -1]).item())
            else:
                p = target_probabilities(logits[0, -1], temperature, top_p, top_k)
                pending = int(mx.random.categorical(mx.log(p)).item())
            prefill_seconds = time.perf_counter() - started
            decode_started = time.perf_counter()
            output.append(pending)
            try:
                yield pending
                while len(output) < max_tokens and pending not in self.tokenizer.eos_token_ids:
                    phase_start = time.perf_counter() if self.profile else 0.0
                    cap = min(self.max_draft, max_tokens - len(output))
                    block = mx.array(
                        [[pending] + [self.drafter.config.mask_token_id] * (self.block_size - 1)]
                    )
                    uniforms = mx.random.uniform(shape=(cap,)) if temperature else None
                    proposals, candidates, q_rows = self.drafter.select_block(
                        block,
                        context,
                        draft_cache,
                        cap=cap,
                        anchor_id=pending,
                        uniforms=uniforms,
                        temperature=temperature,
                    )
                    if self.profile:
                        mx.eval(proposals)
                        phases["draft"] += time.perf_counter() - phase_start
                        phase_start = time.perf_counter()
                    verify_ids = mx.concatenate([mx.array([pending]), proposals]).reshape(1, -1)
                    logits, hidden = self.target.forward(verify_ids, cache, capture=True)
                    if self.profile:
                        mx.eval(logits, hidden)
                        phases["verify"] += time.perf_counter() - phase_start
                        phase_start = time.perf_counter()
                    if temperature == 0:
                        target_ids = mx.argmax(logits[0], axis=-1)
                        matches = (proposals == target_ids[:cap]).astype(mx.int32)
                        accepted_array = mx.cumprod(matches).sum()
                        mx.eval(accepted_array, target_ids, proposals)
                        accepted = int(accepted_array.item())
                        replacement = int(target_ids[accepted].item())
                    else:
                        accepted, replacement = sampled_accept(
                            logits[0], proposals, candidates, q_rows, temperature, top_p, top_k
                        )
                    committed = proposals[:accepted].tolist() + [replacement]
                    if self.profile:
                        phases["accept"] += time.perf_counter() - phase_start
                        phase_start = time.perf_counter()
                    committed = committed[: max_tokens - len(output)]
                    for index, token in enumerate(committed):
                        if token in self.tokenizer.eos_token_ids:
                            committed = committed[: index + 1]
                            break
                    # The final emitted token is the next unprocessed anchor, even
                    # when the output limit or EOS cuts through an accepted block.
                    self.target.rollback(cache, len(committed))
                    context = hidden[:, : len(committed)]
                    if self.profile:
                        mx.eval([item.state for item in cache])
                        phases["rollback"] += time.perf_counter() - phase_start
                    acceptance.append(accepted)
                    for token in committed:
                        output.append(token)
                        yield token
                        if token in self.tokenizer.eos_token_ids:
                            break
                    pending = output[-1]
                self._cached_tokens = tuple(prompt_tokens + output[:-1])
                self._cached_state = (cache, draft_cache, context)
            finally:
                mx.synchronize(stream)
                decode_seconds = time.perf_counter() - decode_started
                self.last_stats = {
                    "prompt_tokens": len(prompt_tokens),
                    "cached_prompt_tokens": cached_count,
                    "prompt_tps": (len(prompt_tokens) - cached_count) / max(prefill_seconds, 1e-9),
                    "prefill_seconds": prefill_seconds,
                    "generation_tokens": len(output),
                    "generation_tps": max(0, len(output) - 1) / max(decode_seconds, 1e-9),
                    "decode_seconds": decode_seconds,
                    "elapsed_seconds": time.perf_counter() - started,
                    "verify_rounds": len(acceptance),
                    "accepted_drafts_per_round": sum(acceptance) / max(len(acceptance), 1),
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
                self.target.records = {}
                if self.profile:
                    self.last_stats["profile_seconds"] = phases
