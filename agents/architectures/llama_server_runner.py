"""Geist runner for a private llama.cpp OpenAI-compatible server."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Generator, Iterator
from typing import Any

import httpx

from agents.architectures.base_runner import (
    BaseRunner,
    GenerationConfig,
    stream_text_until_stop,
    strip_streamed_text,
)
from agents.architectures.llama_server_process import (
    LlamaServerManager,
    get_llama_server_manager,
)
from agents.models.tool_calling import (
    ChatMessage,
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolCall,
    ToolDefinition,
)
from app.services.local_models import LocalModelManager, get_local_model_manager


class LlamaServerRunner(BaseRunner):
    """Resolve a managed GGUF and speak llama-server's OpenAI wire protocol."""

    supports_native_tool_calling = False

    def __init__(
        self,
        *,
        model_manager: LocalModelManager | None = None,
        server_manager: LlamaServerManager | None = None,
    ) -> None:
        self.model_manager = model_manager or get_local_model_manager()
        self.server_manager = server_manager or get_llama_server_manager()
        self.model_id: str | None = None
        self.artifact_id: str | None = None
        self.client: httpx.Client | None = None
        self.base_url: str | None = None
        self.headers: dict[str, str] = {"Content-Type": "application/json"}

    def load(self, model_id: str, device_config: dict[str, Any] | None = None) -> None:
        config = dict(device_config or {})
        artifact_reference = str(config.pop("artifact_id", model_id))
        if config:
            # llama.cpp device/runtime selection is process-wide and controlled
            # by GEIST_LLAMA_ACCELERATION, not untrusted per-request switches.
            unsupported = ", ".join(sorted(config))
            raise ValueError(f"Unsupported llama-server device options: {unsupported}")
        artifact, model_path = self.model_manager.require_installed(artifact_reference)
        connection = self.server_manager.start(model_path, artifact.model_id)
        if self.client is not None:
            self.client.close()
        self.model_id = artifact.model_id
        self.artifact_id = artifact.id
        self.supports_native_tool_calling = bool(getattr(artifact, "supports_tool_calling", False))
        self.base_url = f"{connection.base_url}/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {connection.api_key}",
        }
        request_timeout = float(os.getenv("GEIST_LLAMA_REQUEST_TIMEOUT_SECONDS", "300"))
        if request_timeout <= 0:
            raise ValueError("GEIST_LLAMA_REQUEST_TIMEOUT_SECONDS must be positive")
        self.client = httpx.Client(
            timeout=httpx.Timeout(
                request_timeout,
                connect=min(10.0, request_timeout),
                write=min(30.0, request_timeout),
                pool=min(10.0, request_timeout),
            )
        )

    def _require_client(self) -> httpx.Client:
        if self.client is None or self.base_url is None or self.model_id is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.client

    def _stream_messages(
        self,
        messages: list[dict[str, str | None]],
        generation_config: GenerationConfig,
    ) -> Iterator[str]:
        """Adapt llama-server's native event stream to the shared text stream."""
        request_config = ModelRequestConfig(
            max_tokens=generation_config.max_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            frequency_penalty=generation_config.frequency_penalty,
            presence_penalty=generation_config.presence_penalty,
            stop=(
                [generation_config.stop]
                if isinstance(generation_config.stop, str)
                else generation_config.stop
            ),
        )
        chat_messages = [ChatMessage.from_dict(message) for message in messages]
        for event in self._stream_model_turn_raw(chat_messages, [], request_config):
            if event.kind == "text_delta" and event.text:
                yield event.text

    @staticmethod
    def _provider_tool_name(name: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            return name
        readable = re.sub(r"[^A-Za-z0-9_-]", "_", name).strip("_") or "tool"
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        return f"{readable[:55]}_{digest}"

    @classmethod
    def _tool_name_maps(
        cls,
        tools: list[ToolDefinition],
        messages: list[ChatMessage],
    ) -> tuple[dict[str, str], dict[str, str]]:
        internal_to_provider: dict[str, str] = {}
        provider_to_internal: dict[str, str] = {}
        names = [tool.name for tool in tools]
        for message in messages:
            names.extend(call.name for call in message.tool_calls)
            if message.name:
                names.append(message.name)
        for name in dict.fromkeys(names):
            provider_name = cls._provider_tool_name(name)
            if (
                provider_name in provider_to_internal
                and provider_to_internal[provider_name] != name
            ):
                digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
                provider_name = f"{provider_name[:51]}_{digest}"
            internal_to_provider[name] = provider_name
            provider_to_internal[provider_name] = name
        return internal_to_provider, provider_to_internal

    @staticmethod
    def _arguments(value: str) -> dict[str, Any]:
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError:
            return {"__raw_arguments__": value}
        return parsed if isinstance(parsed, dict) else {"__raw_arguments__": value}

    def _stream_model_turn_raw(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        client = self._require_client()
        internal_to_provider, provider_to_internal = self._tool_name_maps(tools, messages)
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [message.to_openai(internal_to_provider) for message in messages],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "stream": True,
        }
        if config.stop:
            payload["stop"] = config.stop
        if tools:
            payload["tools"] = [tool.to_openai(internal_to_provider[tool.name]) for tool in tools]
            payload["tool_choice"] = "auto"

        text_parts: list[str] = []
        call_parts: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        saw_done = False
        with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(str(content))
                    yield ModelEvent.text_delta(str(content))
                for fallback_index, tool_delta in enumerate(delta.get("tool_calls") or []):
                    index = int(tool_delta.get("index", fallback_index))
                    current = call_parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    current["id"] += tool_delta.get("id") or ""
                    function = tool_delta.get("function") or {}
                    current["name"] += function.get("name") or ""
                    current["arguments"] += function.get("arguments") or ""

        if not saw_done and finish_reason is None:
            raise RuntimeError("llama-server stream ended without a terminal marker")
        calls = [
            ToolCall(
                id=value["id"] or f"toolcall_{uuid.uuid4().hex}",
                name=provider_to_internal.get(value["name"], value["name"]),
                arguments=self._arguments(value["arguments"]),
            )
            for _index, value in sorted(call_parts.items())
        ]
        yield ModelEvent.turn_complete(
            ModelTurn(
                text="".join(text_parts),
                tool_calls=calls,
                finish_reason=finish_reason,
            )
        )

    def stream_model_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        """Apply shared text semantics while preserving native tool events."""
        completed_turn: ModelTurn | None = None
        raw_events = self._stream_model_turn_raw(messages, tools, config)

        def text_chunks() -> Generator[str, None, None]:
            nonlocal completed_turn
            for event in raw_events:
                if event.kind == "text_delta" and event.text:
                    yield event.text
                elif event.kind == "turn_complete":
                    completed_turn = event.turn

        normalized_parts: list[str] = []
        text_stream = text_chunks()
        try:
            for text_delta in strip_streamed_text(stream_text_until_stop(text_stream, config.stop)):
                normalized_parts.append(text_delta)
                yield ModelEvent.text_delta(text_delta)
        finally:
            text_stream.close()

        yield ModelEvent.turn_complete(
            ModelTurn(
                text="".join(normalized_parts),
                tool_calls=(completed_turn.tool_calls if completed_turn else []),
                finish_reason=(completed_turn.finish_reason if completed_turn else "stop"),
            )
        )

    def cleanup(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        self.server_manager.stop()
