"""Adapter from Geist's completion contract to the optional mlx-lm runtime."""

import importlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

from agents.architectures.chat_template_tools import (
    ToolResponseStream,
    build_tool_payload,
    parse_tool_response,
    strip_inkling_response_markup,
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
DFLASH_MIN_TOKENS = 32
MAX_RETAINED_PREFIX_TOKENS = 32768
INKLING2_MODEL_TYPES = frozenset({"inkling2", "inkling2_loop"})
logger = logging.getLogger(__name__)


def _is_qwen3_model(model_id: str) -> bool:
    return re.search(r"(?<![a-z0-9])qwen[-_ ]?3(?:[._]\d+)?(?!\d)", model_id, re.I) is not None


def _is_inkling2_model(model_id: str) -> bool:
    return re.search(r"(?<![a-z0-9])inkling[-_ ]?2(?![0-9])", model_id, re.I) is not None


def _read_local_model_type(path_or_id: str) -> str | None:
    config_path = Path(path_or_id) / "config.json"
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    model_type = data.get("model_type")
    return model_type if isinstance(model_type, str) else None


def _load_model_and_tokenizer(path_or_id: str) -> tuple[Any, Any]:
    """Load an MLX model and tokenizer, routing local Inkling 2 snapshots to Geist's architecture.

    Inkling 2 detection inspects ``config.json`` inside a local snapshot directory
    (which ``MLXLlamaRunner`` supplies via ``weights_dir`` from the managed store).
    """
    model_type = _read_local_model_type(path_or_id)
    if model_type in INKLING2_MODEL_TYPES:
        from mlx_lm.utils import load_model, load_tokenizer

        from agents.architectures.inkling2 import Model as Inkling2Model
        from agents.architectures.inkling2 import ModelArgs as Inkling2ModelArgs

        def _get_inkling2_classes(config: dict[str, Any]) -> tuple[type[Any], type[Any]]:
            if config.get("model_type") in INKLING2_MODEL_TYPES:
                return Inkling2Model, Inkling2ModelArgs
            from mlx_lm.utils import _get_classes

            return _get_classes(config)

        model_path = Path(path_or_id)
        model, config = load_model(
            model_path,
            lazy=False,
            get_model_classes=_get_inkling2_classes,
        )
        tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id"))
        return model, tokenizer

    from mlx_lm import load

    return load(path_or_id)


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


class _InklingStreamFilter:
    """Incrementally extract plain text from Inkling 2 control-token streams."""

    _TEXT_MARKERS = (
        "<|content_text|>",
        "<|content_tool_error|>",
        "<|content_invoke_tool_text|>",
    )
    _END_MARKERS = (
        "<|end_message|>",
        "<|content_model_end_sampling|>",
    )
    _ALL_MARKERS = (
        _TEXT_MARKERS
        + _END_MARKERS
        + (
            "<|message_model|>",
            "<|content_thinking|>",
            "<|content_invoke_tool_json|>",
        )
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._in_text = False
        self._saw_markup = False

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        output: list[str] = []
        while self._buffer:
            if not self._in_text:
                text_pos = -1
                text_marker = ""
                for marker in self._TEXT_MARKERS:
                    idx = self._buffer.find(marker)
                    if idx != -1 and (text_pos == -1 or idx < text_pos):
                        text_pos = idx
                        text_marker = marker
                if text_pos != -1:
                    if text_pos and not self._saw_markup:
                        output.append(self._buffer[:text_pos])
                    self._saw_markup = True
                    self._in_text = True
                    self._buffer = self._buffer[text_pos + len(text_marker) :]
                    continue
                marker_pos = -1
                for marker in self._ALL_MARKERS:
                    idx = self._buffer.find(marker)
                    if idx != -1 and (marker_pos == -1 or idx < marker_pos):
                        marker_pos = idx
                if marker_pos != -1:
                    if marker_pos and not self._saw_markup:
                        output.append(self._buffer[:marker_pos])
                    self._saw_markup = True
                    retained = _stop_prefix_length(self._buffer, self._TEXT_MARKERS)
                    self._buffer = self._buffer[len(self._buffer) - retained :] if retained else ""
                    break
                if self._saw_markup:
                    retained = _stop_prefix_length(self._buffer, self._TEXT_MARKERS)
                    self._buffer = self._buffer[len(self._buffer) - retained :] if retained else ""
                    break
                # Wait briefly if the buffer may be a partial leading control tag.
                if len(self._buffer) < 32 and _stop_prefix_length(
                    self._buffer, self._ALL_MARKERS
                ) == len(self._buffer):
                    break
                retained = _stop_prefix_length(self._buffer, self._ALL_MARKERS)
                emit_len = len(self._buffer) - retained
                if emit_len:
                    output.append(self._buffer[:emit_len])
                    self._buffer = self._buffer[emit_len:]
                break

            end_pos = -1
            end_marker = ""
            for marker in self._END_MARKERS:
                idx = self._buffer.find(marker)
                if idx != -1 and (end_pos == -1 or idx < end_pos):
                    end_pos = idx
                    end_marker = marker
            if end_pos != -1:
                if end_pos:
                    output.append(self._buffer[:end_pos])
                self._in_text = False
                self._buffer = self._buffer[end_pos + len(end_marker) :]
                continue
            retained = _stop_prefix_length(self._buffer, self._END_MARKERS)
            emit_len = len(self._buffer) - retained
            if emit_len:
                output.append(self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
            break
        return "".join(output)

    def flush(self) -> str:
        if not self._buffer:
            return ""
        if self._in_text or not self._saw_markup:
            rest = strip_inkling_response_markup(self._buffer)
            self._buffer = ""
            return rest
        self._buffer = ""
        return ""


def _configure_thread_local_generation_stream() -> None:
    mx = importlib.import_module("mlx.core")
    generate_module = importlib.import_module("mlx_lm.generate")
    generate_module.generation_stream = mx.new_thread_local_stream(mx.default_device())


def _prefill_step_size() -> int:
    raw = os.environ.get("GEIST_MLX_PREFILL_STEP_SIZE", "2048")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("GEIST_MLX_PREFILL_STEP_SIZE must be an integer") from error
    if value <= 0:
        raise ValueError("GEIST_MLX_PREFILL_STEP_SIZE must be greater than zero")
    return value


def _cache_matches_tokens(cache, count: int) -> bool:
    offsets = [item.offset for item in cache if isinstance(getattr(item, "offset", None), int)]
    if not offsets:
        logger.debug("Prefix cache reuse disabled: cache exposes no verifiable token offset")
    return bool(offsets) and all(offset == count for offset in offsets)


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
            importlib.import_module("mlx_lm")
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
        self.model, self.tokenizer = _load_model_and_tokenizer(weights_dir or model_id)
        self.last_stats: dict[str, int | float | str | None] = {}
        self.prefill_step_size = _prefill_step_size()
        # Starlette may resume/close a synchronous stream on a different worker.
        self._generation_lock = threading.Lock()
        self._prompt_cache = None
        self._cached_tokens: tuple[int, ...] = ()
        self._dflash = None
        self._dflash_checked = False
        self._small_m_wrappers = []
        self.small_m_tuning = []
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

    def _prepare_dflash(self) -> None:
        if getattr(self, "_dflash_checked", False):
            return
        from agents.architectures.llama.dflash_artifact import find_dflash_path

        path = find_dflash_path(self.model_id)
        if path is not None:
            from agents.architectures.llama.dflash_backend import DFlashDecoder, load_drafter
            from agents.architectures.llama.qwen_small_m import install_small_m, tune_small_m

            self._small_m_wrappers = []
            try:
                drafter = load_drafter(str(path), self.model)
                decoder = DFlashDecoder(
                    self.model,
                    self.tokenizer,
                    drafter,
                    prefill_step_size=self.prefill_step_size,
                    adaptive=os.environ.get("GEIST_MLX_DFLASH_ADAPTIVE", "on").casefold() != "off",
                )
                self._small_m_wrappers = install_small_m(self.model)
                # Rebind after wrapping so the drafter shares the wrapped head.
                drafter.bind(self.model)
                self._small_m_wrappers += install_small_m(drafter)
                self.small_m_tuning = tune_small_m(self._small_m_wrappers)
                self._dflash = decoder
                logger.info("Enabled MLX DFlash 2 with qualified Metal kernels for Qwen 3.8")
            except Exception as error:
                for wrapper in self._small_m_wrappers:
                    wrapper.enabled = False
                if os.environ.get("GEIST_MLX_DFLASH", "auto").casefold() == "on":
                    raise
                logger.warning("DFlash initialization failed; using ordinary MLX: %s", error)
        self._dflash_checked = True

    def _stream_dflash(self, prompt_tokens: list[int]) -> Iterator[str]:
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        generator = self._dflash.generate(
            prompt_tokens,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=QWEN3_TOP_K if _is_qwen3_model(self.model_id) else 0,
        )
        try:
            with closing(generator):
                for token in generator:
                    if token not in self.tokenizer.eos_token_ids:
                        detokenizer.add_token(token)
                        segment = detokenizer.last_segment
                        if segment:
                            yield segment
            detokenizer.finalize()
            segment = detokenizer.last_segment
            if segment:
                yield segment
        finally:
            self.last_stats = {"implementation": "mlx_dflash", **self._dflash.last_stats}

    def _is_inkling2(self) -> bool:
        model = getattr(self, "model", None)
        model_id = getattr(self, "model_id", "") or ""
        return (
            _is_inkling2_model(model_id)
            or getattr(model, "model_type", None) in INKLING2_MODEL_TYPES
        )

    def _template_options(self) -> dict[str, Any]:
        template_options = dict(self.chat_template_kwargs)
        if _is_qwen3_model(self.model_id):
            template_options.setdefault("enable_thinking", False)
        if self._is_inkling2():
            template_options.setdefault("reasoning_effort", "none")
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
        pending_text = ""
        stopped = False
        inkling_filter = _InklingStreamFilter() if (self._is_inkling2() and not tools) else None
        response_stream = self._stream_prompt(prompt, sampler, logits_processors or None)
        try:
            for segment in response_stream:
                chunk = inkling_filter.feed(segment) if inkling_filter is not None else segment
                if not chunk:
                    continue
                pending_text += chunk
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

        if inkling_filter is not None and not stopped:
            tail = inkling_filter.flush()
            if tail:
                pending_text += tail
                stop_index = _first_stop_index(pending_text, stops)
                if stop_index is not None:
                    if stop_index:
                        yield pending_text[:stop_index]
                    pending_text = ""
                    stopped = True

        if pending_text and not stopped:
            yield pending_text

    def _stream_prompt(self, prompt: str, sampler, logits_processors=None) -> Iterator[str]:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        add_special_tokens = self.tokenizer.bos_token is None or not prompt.startswith(
            self.tokenizer.bos_token
        )
        prompt_tokens = list(self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
        with self._generation_lock:
            if self.max_new_tokens >= DFLASH_MIN_TOKENS and not logits_processors:
                self._prepare_dflash()
            use_dflash = (
                getattr(self, "_dflash", None) is not None
                and self.max_new_tokens >= DFLASH_MIN_TOKENS
                and not logits_processors
            )
            for wrapper in getattr(self, "_small_m_wrappers", []):
                wrapper.enabled = use_dflash
            if use_dflash:
                yield from self._stream_dflash(prompt_tokens)
                return
            cached_count = len(self._cached_tokens)
            cache_hit = (
                self._prompt_cache is not None
                and _cache_matches_tokens(self._prompt_cache, cached_count)
                and cached_count < len(prompt_tokens)
                and tuple(prompt_tokens[:cached_count]) == self._cached_tokens
            )
            if cache_hit:
                prompt_cache = self._prompt_cache
                suffix = prompt_tokens[cached_count:]
            else:
                prompt_cache = make_prompt_cache(self.model)
                suffix = prompt_tokens
                cached_count = 0

            started = time.perf_counter()
            final_response = None
            output_tokens = []
            responses = stream_generate(
                self.model,
                self.tokenizer,
                mx.array(suffix),
                max_tokens=self.max_new_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                prefill_step_size=self.prefill_step_size,
            )
            try:
                for response in responses:
                    final_response = response
                    output_tokens.append(int(response.token))
                    if response.text:
                        yield response.text
            except BaseException:
                self._prompt_cache = None
                self._cached_tokens = ()
                raise
            finally:
                close = getattr(responses, "close", None)
                if callable(close):
                    close()

            elapsed = time.perf_counter() - started
            cached_tokens = tuple(prompt_tokens + output_tokens)
            # In mlx-lm 0.31.3 the lookahead step consumes the token before it
            # is yielded. Check actual KV offsets rather than assuming this
            # contract survives a future dependency change.
            if len(cached_tokens) <= MAX_RETAINED_PREFIX_TOKENS and _cache_matches_tokens(
                prompt_cache, len(cached_tokens)
            ):
                self._prompt_cache = prompt_cache
                self._cached_tokens = cached_tokens
            else:
                self._prompt_cache = None
                self._cached_tokens = ()
            if final_response is not None:
                self.last_stats = {
                    "implementation": "mlx_lm",
                    "prompt_tokens": len(prompt_tokens),
                    "cached_prompt_tokens": cached_count,
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

    def cleanup(self) -> None:
        with self._generation_lock:
            self._prompt_cache = None
            self._cached_tokens = ()
            self._dflash = None
            for wrapper in self._small_m_wrappers:
                wrapper.enabled = False
            self._small_m_wrappers = []
            self.small_m_tuning = []
            self.model = None
            self.tokenizer = None
            self.supports_native_tool_calling = False

    def complete_messages(
        self,
        messages: list[dict[str, str | None]],
    ) -> list[dict[str, str]]:
        response = "".join(self.stream_messages(messages)).strip()
        if self._is_inkling2():
            response = strip_inkling_response_markup(response)
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
        responses = self.stream_messages(payload.messages, payload.tools)
        if self._is_inkling2():
            stream_filter = _InklingStreamFilter() if payload.tools else None
            raw_parts: list[str] = []
            try:
                for segment in responses:
                    raw_parts.append(segment)
                    text = stream_filter.feed(segment) if stream_filter is not None else segment
                    if text:
                        yield ModelEvent.text_delta(text)
                if stream_filter is not None:
                    tail = stream_filter.flush()
                    if tail:
                        yield ModelEvent.text_delta(tail)
            finally:
                close = getattr(responses, "close", None)
                if callable(close):
                    close()
            raw_response = "".join(raw_parts).strip()
            turn = parse_tool_response(
                raw_response,
                provider_to_internal=payload.provider_to_internal,
                tools=payload.tools,
            )
            yield ModelEvent.turn_complete(turn)
            return
        parser = ToolResponseStream(payload.provider_to_internal, payload.tools)
        try:
            for segment in responses:
                text = parser.feed(segment)
                if text:
                    yield ModelEvent.text_delta(text)
            tail, turn = parser.finish()
            if tail:
                yield ModelEvent.text_delta(tail)
            yield ModelEvent.turn_complete(turn)
        finally:
            close = getattr(responses, "close", None)
            if callable(close):
                close()
