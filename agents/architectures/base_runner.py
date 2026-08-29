"""
Base runner abstract class for all inference backends.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

from agents.models.tool_calling import (
    ChatMessage,
    ModelEvent,
    ModelRequestConfig,
    ModelTurn,
    ToolDefinition,
)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None


def stream_text_until_stop(
    chunks: Iterable[str],
    stop: str | list[str] | None,
) -> Iterator[str]:
    """Yield chunks without exposing a configured stop sequence."""
    stop_sequences = [stop] if isinstance(stop, str) else list(stop or [])
    stop_sequences = [sequence for sequence in stop_sequences if sequence]
    if not stop_sequences:
        yield from (chunk for chunk in chunks if chunk)
        return

    buffered = ""
    for chunk in chunks:
        if not chunk:
            continue
        buffered += chunk
        stop_positions = [
            position for sequence in stop_sequences if (position := buffered.find(sequence)) >= 0
        ]
        if stop_positions:
            text = buffered[: min(stop_positions)]
            if text:
                yield text
            return

        retained_characters = max(
            (
                prefix_length
                for sequence in stop_sequences
                for prefix_length in range(1, len(sequence))
                if buffered.endswith(sequence[:prefix_length])
            ),
            default=0,
        )
        safe_length = len(buffered) - retained_characters
        if safe_length > 0:
            yield buffered[:safe_length]
            buffered = buffered[safe_length:]

    if buffered:
        yield buffered


def strip_streamed_text(chunks: Iterable[str]) -> Iterator[str]:
    """Strip only response-edge whitespace without buffering the full response."""
    leading = True
    trailing_whitespace = ""
    for chunk in chunks:
        if not chunk:
            continue
        if leading:
            chunk = chunk.lstrip()
            if not chunk:
                continue
            leading = False

        text = trailing_whitespace + chunk
        stripped = text.rstrip()
        trailing_whitespace = text[len(stripped) :]
        if stripped:
            yield stripped


class BaseRunner(ABC):
    """Abstract base class for all inference runners."""

    supports_native_tool_calling = False

    @abstractmethod
    def load(self, model_id: str, device_config: dict[str, Any] | None = None) -> None:
        """
        Load the model and prepare for inference.

        Args:
            model_id: Identifier for the model to load
            device_config: Optional device configuration (GPU, CPU, etc.)
        """
        pass

    def generate(
        self, prompt: str, generation_config: GenerationConfig
    ) -> dict[str, Any] | list[dict[str, str]]:
        """Render a plain prompt as one user chat message and collect the stream."""
        return self.complete("", prompt, generation_config)

    def complete(
        self, system_prompt: str, user_prompt: str, generation_config: GenerationConfig
    ) -> list[dict[str, str]]:
        """Collect system and user prompts through the canonical message stream."""
        messages: list[dict[str, str | None]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.complete_messages(messages, generation_config)

    def complete_messages(
        self,
        messages: list[dict[str, str | None]],
        generation_config: GenerationConfig,
    ) -> list[dict[str, str]]:
        """Collect the canonical message stream into the legacy response shape."""
        response = "".join(self.stream_messages(messages, generation_config))
        user_prompt = next(
            (
                str(message.get("content") or "")
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]

    @abstractmethod
    def _stream_messages(
        self,
        messages: list[dict[str, str | None]],
        generation_config: GenerationConfig,
    ) -> Iterator[str]:
        """Yield raw incremental text from the selected inference runtime."""
        raise NotImplementedError

    def stream_messages(
        self,
        messages: list[dict[str, str | None]],
        generation_config: GenerationConfig,
    ) -> Iterator[str]:
        """Apply shared response semantics to the runtime's incremental output."""
        yield from strip_streamed_text(
            stream_text_until_stop(
                self._stream_messages(messages, generation_config),
                generation_config.stop,
            )
        )

    def stream_model_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        config: ModelRequestConfig,
    ) -> Iterator[ModelEvent]:
        """Adapt the canonical text stream to Geist's model-event contract."""
        if tools:
            raise ValueError(f"{type(self).__name__} does not support native tool calling")
        generation_config = GenerationConfig(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            stop=config.stop,
        )
        text_parts: list[str] = []
        structured_messages: list[dict[str, str | None]] = []
        for message in messages:
            provider_message = message.to_openai()
            provider_message["content"] = message.content or ""
            if message.name:
                provider_message["name"] = message.name
            structured_messages.append(cast(dict[str, str | None], provider_message))
        for text_delta in self.stream_messages(structured_messages, generation_config):
            text_parts.append(text_delta)
            yield ModelEvent.text_delta(text_delta)
        yield ModelEvent.turn_complete(ModelTurn(text="".join(text_parts), finish_reason="stop"))

    def cleanup(self) -> None:  # noqa: B027 - optional hook, runners override as needed
        """
        Clean up resources (optional override).
        Default implementation does nothing.
        """
        pass
