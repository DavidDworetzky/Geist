"""Switchable MLX Llama runner."""

import json
import logging
import os
import re
from collections.abc import Iterator
from typing import Any, Protocol

from agents.models.llama_completion import LlamaCompletion
from agents.models.tool_calling import (
    ChatMessage,
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolDefinition,
)

from .base_runner import BaseRunner, GenerationConfig


logger = logging.getLogger(__name__)


class _MLXBackend(Protocol):
    """Shared runtime contract implemented by both MLX backends."""

    max_new_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: str | list[str] | None

    def complete(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]: ...

    def complete_messages(
        self,
        messages: list[dict[str, str | None]],
    ) -> list[dict[str, str]]: ...


class MLXLlamaRunner(BaseRunner):
    """Run Llama through Geist's manual MLX code or the mlx-lm adapter."""

    IMPLEMENTATIONS = {"manual", "mlx_lm"}

    def __init__(self):
        self.llama: _MLXBackend | None = None
        self.model_id: str | None = None
        self.weights_dir: str | None = None
        self.implementation: str | None = None
        self._unsupported_generation_controls_warned = False
        self.supports_native_tool_calling = False

    @staticmethod
    def _manual_implementation_supports(model_id: str, weights_dir: str) -> bool:
        """The hand-written kernel only implements the Llama architecture."""
        from agents.model_catalog import infer_model_spec

        config_path = os.path.join(weights_dir, "config.json")
        try:
            with open(config_path, encoding="utf-8") as config_file:
                model_type = json.load(config_file).get("model_type")
        except (AttributeError, json.JSONDecodeError, OSError):
            model_type = None
        if isinstance(model_type, str):
            return model_type.casefold() == "llama"

        spec = infer_model_spec(model_id)
        if spec is not None and spec.family == "llama":
            return True
        return (
            re.search(
                r"(?:^|[/_-])[a-z]*llama(?:[-_./]|$)",
                model_id,
                re.IGNORECASE,
            )
            is not None
        )

    @staticmethod
    def _resolve_weights_dir(model_id: str, device_config: dict[str, Any]) -> str:
        configured = device_config.get("weights_dir")
        if configured is not None:
            if not isinstance(configured, str):
                raise TypeError("weights_dir must be a string")
            return os.path.expanduser(configured)

        artifact_reference = device_config.get("artifact_id")
        if artifact_reference is not None:
            if not isinstance(artifact_reference, str):
                raise TypeError("artifact_id must be a string")
            return MLXLlamaRunner._resolve_managed_artifact(artifact_reference)

        local_root = os.environ.get("LOCAL_WEIGHTS_DIR")
        if local_root:
            local_root = os.path.expanduser(local_root)
            if os.path.exists(os.path.join(local_root, "config.json")):
                return local_root
            return os.path.join(local_root, "llama_3_1")

        return MLXLlamaRunner._resolve_managed_artifact(model_id)

    @staticmethod
    def _resolve_managed_artifact(artifact_reference: str) -> str:
        from app.services.local_models import get_local_model_manager

        artifact, installed_path = get_local_model_manager().require_installed(artifact_reference)
        if artifact.backend != "mlx_llama":
            raise ValueError(
                f"Managed artifact {artifact.id} is not compatible with the MLX runner"
            )
        return str(installed_path)

    def load(self, model_id: str, device_config: dict[str, Any] | None = None) -> None:
        """Load the selected implementation and propagate the requested model path."""
        device_config = device_config or {}
        requested = device_config.get("implementation")
        if requested is None:
            requested = os.environ.get("GEIST_MLX_IMPLEMENTATION")
        if requested is not None and not isinstance(requested, str):
            raise TypeError("MLX implementation must be a string")
        implementation = (
            requested.strip().lower().replace("-", "_") if requested is not None else None
        )
        if implementation is not None and implementation not in self.IMPLEMENTATIONS:
            choices = ", ".join(sorted(self.IMPLEMENTATIONS))
            raise ValueError(
                f"Unknown MLX implementation '{requested}'. Expected one of: {choices}."
            )

        weights_dir = self._resolve_weights_dir(model_id, device_config)
        manual_supported = self._manual_implementation_supports(model_id, weights_dir)
        if implementation is None:
            implementation = "manual" if manual_supported else "mlx_lm"
        if implementation == "manual" and not manual_supported:
            raise ValueError(
                "The manual MLX implementation only supports Llama models; "
                f"use implementation='mlx_lm' for {model_id}."
            )

        if implementation == "manual":
            from agents.architectures.llama.llama_mlx import LlamaMLX

            backend: _MLXBackend = LlamaMLX(
                max_new_tokens=16,
                model_id=model_id,
                weights_dir=weights_dir,
            )
        else:
            from agents.architectures.llama.mlx_lm_backend import MLXLMBackend

            backend_kwargs: dict[str, Any] = {
                "max_new_tokens": 16,
                "model_id": model_id,
                "weights_dir": weights_dir,
            }
            chat_template_kwargs = device_config.get("chat_template_kwargs")
            if chat_template_kwargs is not None:
                if not isinstance(chat_template_kwargs, dict):
                    raise TypeError("chat_template_kwargs must be a dictionary")
                backend_kwargs["chat_template_kwargs"] = chat_template_kwargs
            backend = MLXLMBackend(
                **backend_kwargs,
            )

        self.llama = backend
        self.model_id = model_id
        self.weights_dir = weights_dir
        self.implementation = implementation
        self._unsupported_generation_controls_warned = False
        self.supports_native_tool_calling = bool(
            implementation == "mlx_lm" and getattr(backend, "supports_native_tool_calling", False)
        )

        logger.info(
            "MLX Llama runner loaded implementation=%s model=%s weights=%s",
            self.implementation,
            model_id,
            self.weights_dir or "Hugging Face",
        )

    def _apply_generation_config(self, generation_config: GenerationConfig) -> _MLXBackend:
        backend = self.llama
        if backend is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        backend.max_new_tokens = generation_config.max_tokens
        backend.temperature = generation_config.temperature
        backend.top_p = generation_config.top_p
        if self.implementation == "mlx_lm":
            backend.frequency_penalty = generation_config.frequency_penalty
            backend.presence_penalty = generation_config.presence_penalty
            backend.stop = generation_config.stop
        elif (
            generation_config.frequency_penalty
            or generation_config.presence_penalty
            or generation_config.stop
        ) and not self._unsupported_generation_controls_warned:
            logger.warning(
                "The manual MLX implementation ignores frequency_penalty, "
                "presence_penalty, and stop; select implementation='mlx_lm' "
                "to use those controls."
            )
            self._unsupported_generation_controls_warned = True
        return backend

    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> list[dict[str, str]]:
        return self.complete("", prompt, generation_config)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_config: GenerationConfig,
    ) -> list[dict[str, str]]:
        backend = self._apply_generation_config(generation_config)
        return backend.complete(
            system_prompt=system_prompt or "You are a helpful assistant.",
            user_prompt=user_prompt,
        )

    def complete_messages(
        self,
        messages: list[dict[str, str | None]],
        generation_config: GenerationConfig,
    ) -> list[dict[str, str]]:
        backend = self._apply_generation_config(generation_config)
        return backend.complete_messages(messages)

    def stream_model_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        """Run a structured turn through mlx-lm, keeping manual MLX text-only."""

        if tools and not self.supports_native_tool_calling:
            raise ValueError(f"Model {self.model_id} does not support native tool calling")
        backend = self._apply_generation_config(
            GenerationConfig(
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop,
            )
        )
        native_stream = getattr(backend, "stream_model_turn", None)
        if callable(native_stream):
            yield from native_stream(messages, tools, config)
            return

        structured_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        stream_messages = getattr(backend, "stream_messages", None)
        if callable(stream_messages):
            segments = []
            responses = stream_messages(structured_messages)
            try:
                for segment in responses:
                    if segment:
                        segments.append(segment)
                        yield ModelEvent.text_delta(segment)
            finally:
                close = getattr(responses, "close", None)
                if callable(close):
                    close()
            yield ModelEvent.turn_complete(
                ModelTurn(text="".join(segments).strip(), finish_reason="stop")
            )
            return
        completion = LlamaCompletion.from_dict(backend.complete_messages(structured_messages))
        text = next(
            (message.content for message in completion.messages if message.role == "assistant"),
            "",
        )
        if text:
            yield ModelEvent.text_delta(text)
        yield ModelEvent.turn_complete(ModelTurn(text=text, finish_reason="stop"))

    def cleanup(self) -> None:
        cleanup = getattr(self.llama, "cleanup", None)
        if callable(cleanup):
            cleanup()
        self.llama = None
        self.supports_native_tool_calling = False
        logger.info("MLX Llama runner cleaned up")
