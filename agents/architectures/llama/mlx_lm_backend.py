"""Adapter from Geist's completion contract to the optional mlx-lm runtime."""

import importlib
import logging
import re
import time
from collections.abc import Iterator
from typing import Any

from agents.architectures.chat_template_tools import (
    build_tool_payload,
    parse_tool_response,
    tokenizer_supports_tools,
)
from agents.model_catalog import infer_model_spec
from agents.models.llama_completion import strings_to_message_dict
from agents.models.tool_calling import (
    ChatMessage,
    ModelEvent,
    ModelRequestConfig,
    ToolDefinition,
)


QWEN3_TOP_K = 20
logger = logging.getLogger(__name__)


def _is_qwen3_model(model_id: str) -> bool:
    return re.search(r"(?<![a-z0-9])qwen[-_ ]?3(?:[._]\d+)?(?!\d)", model_id, re.I) is not None


def _normalize_stops(stop: str | list[str] | None) -> tuple[str, ...]:
    if isinstance(stop, str):
        stop = [stop]
    return tuple(dict.fromkeys(value for value in stop or [] if value))


def _first_stop_index(text: str, stops: tuple[str, ...]) -> int | None:
    positions = (text.find(stop) for stop in stops)
    return min((position for position in positions if position >= 0), default=None)


def _stop_prefix_length(text: str, stops: tuple[str, ...]) -> int:
    """Retain a suffix that may become a stop sequence in the next text chunk."""
    retained = 0
    for stop in stops:
        maximum = min(len(text), len(stop) - 1)
        for length in range(maximum, 0, -1):
            if text.endswith(stop[:length]):
                retained = max(retained, length)
                break
    return retained


def _configure_thread_local_generation_stream() -> None:
    mx = importlib.import_module("mlx.core")
    generate_module = importlib.import_module("mlx_lm.generate")
    generate_module.generation_stream = mx.new_thread_local_stream(mx.default_device())


class MLXLMBackend:
    """Load and generate with mlx-lm while matching ``LlamaMLX.complete``."""

    def __init__(
        self,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 1.0,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        weights_dir: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "The mlx_lm implementation requires the pinned mlx-lm dependency."
            ) from exc

        _configure_thread_local_generation_stream()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.stop: str | list[str] | None = None
        self.model_id = model_id
        self.weights_dir = weights_dir
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.model, self.tokenizer = load(weights_dir or model_id)
        self.last_stats: dict[str, float] = {}
        self._unsupported_penalties_warned = False
        model_spec = infer_model_spec(model_id)
        self.supports_native_tool_calling = bool(
            model_spec
            and model_spec.supports_function_calling
            and tokenizer_supports_tools(
                self.tokenizer,
                template_options=self._template_options(),
            )
        )

    def _template_options(self) -> dict[str, Any]:
        template_options = dict(self.chat_template_kwargs)
        if _is_qwen3_model(self.model_id):
            template_options.setdefault("enable_thinking", False)
        return template_options

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self._build_messages_prompt(messages)

    def _build_messages_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        normalized = [dict(message) for message in messages]
        template_options = self._template_options()
        template_options.update(
            {
                "tokenize": False,
                "add_generation_prompt": True,
            }
        )
        if tools:
            template_options["tools"] = tools
        return self.tokenizer.apply_chat_template(
            normalized,
            **template_options,
        )

    def stream_text(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Yield decoded text segments and retain mlx-lm timing statistics."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        yield from self.stream_messages(messages)

    def stream_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Yield decoded text for a structured conversation."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        prompt = self._build_messages_prompt(messages, tools)
        sampler_options: dict[str, float | int] = {
            "temp": self.temperature,
            "top_p": self.top_p,
        }
        if _is_qwen3_model(self.model_id):
            sampler_options["top_k"] = QWEN3_TOP_K
        sampler = make_sampler(**sampler_options)
        if (self.presence_penalty or self.frequency_penalty) and not getattr(
            self, "_unsupported_penalties_warned", False
        ):
            logger.warning(
                "mlx-lm does not expose OpenAI-style presence/frequency penalties; "
                "these controls are ignored for this runtime"
            )
            self._unsupported_penalties_warned = True
        logits_processors = make_logits_processors()
        stops = _normalize_stops(self.stop)
        started = time.perf_counter()
        final_response = None
        pending_text = ""
        stopped = False
        response_stream = stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
            logits_processors=logits_processors or None,
        )
        try:
            for response in response_stream:
                final_response = response
                pending_text += response.text
                stop_index = _first_stop_index(pending_text, stops)
                if stop_index is not None:
                    if stop_index:
                        yield pending_text[:stop_index]
                    stopped = True
                    break

                retained = _stop_prefix_length(pending_text, stops)
                emit_length = len(pending_text) - retained
                if emit_length:
                    yield pending_text[:emit_length]
                    pending_text = pending_text[emit_length:]
        finally:
            close = getattr(response_stream, "close", None)
            if callable(close):
                close()

        if pending_text and not stopped:
            yield pending_text

        elapsed = time.perf_counter() - started
        if final_response is not None:
            self.last_stats = {
                "prompt_tokens": int(final_response.prompt_tokens),
                "prompt_tps": float(final_response.prompt_tps),
                "generation_tokens": int(final_response.generation_tokens),
                "generation_tps": float(final_response.generation_tps),
                "peak_memory_gb": float(final_response.peak_memory),
                "elapsed_seconds": elapsed,
            }

    def complete(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.complete_messages(messages)

    def complete_messages(
        self,
        messages: list[dict[str, str | None]],
    ) -> list[dict[str, str]]:
        response = "".join(self.stream_messages(messages)).strip()
        user_prompt = next(
            (
                message.get("content") or ""
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        return strings_to_message_dict(user_prompt, response)

    def stream_model_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        """Generate and normalize one mlx-lm structured model turn."""

        del config  # Generation controls are applied by MLXLlamaRunner.
        if tools and not self.supports_native_tool_calling:
            raise ValueError(f"Model {self.model_id} does not support native tool calling")
        payload = build_tool_payload(messages, tools)
        response = "".join(self.stream_messages(payload.messages, payload.tools)).strip()
        turn = parse_tool_response(
            response,
            provider_to_internal=payload.provider_to_internal,
        )
        if turn.text:
            yield ModelEvent.text_delta(turn.text)
        yield ModelEvent.turn_complete(turn)
