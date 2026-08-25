"""
Base runner abstract class for all inference backends.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any


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

    # Retain enough trailing text to recognize a delimiter split across two
    # model chunks. Everything before that suffix is safe to expose to chat.
    retained_characters = max(len(sequence) for sequence in stop_sequences) - 1
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

        safe_length = len(buffered) - retained_characters
        if safe_length > 0:
            yield buffered[:safe_length]
            buffered = buffered[safe_length:]

    if buffered:
        yield buffered


class BaseRunner(ABC):
    """Abstract base class for all inference runners."""

    @abstractmethod
    def load(self, model_id: str, device_config: dict[str, Any] | None = None) -> None:
        """
        Load the model and prepare for inference.

        Args:
            model_id: Identifier for the model to load
            device_config: Optional device configuration (GPU, CPU, etc.)
        """
        pass

    @abstractmethod
    def generate(
        self, prompt: str, generation_config: GenerationConfig
    ) -> dict[str, Any] | list[dict[str, str]]:
        """
        Generate text based on the given prompt.

        Args:
            prompt: Input text prompt
            generation_config: Configuration for generation parameters

        Returns:
            Dictionary containing generated text and metadata
        """
        pass

    @abstractmethod
    def complete(
        self, system_prompt: str, user_prompt: str, generation_config: GenerationConfig
    ) -> list[dict[str, str]]:
        """
        Complete a conversation with system and user prompts.

        Args:
            system_prompt: System instructions
            user_prompt: User input
            generation_config: Configuration for generation parameters

        Returns:
            Dictionary containing completion and metadata
        """
        pass

    def complete_messages(
        self,
        messages: list[dict[str, str | None]],
        generation_config: GenerationConfig,
    ) -> list[dict[str, str]]:
        """Complete structured chat messages with a text-only compatibility fallback."""
        system_prompt = "\n\n".join(
            message.get("content") or "" for message in messages if message.get("role") == "system"
        )
        conversation = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            label = "Assistant" if role == "assistant" else "User"
            if role == "tool":
                label = "Tool"
            conversation.append(f"{label}: {message.get('content') or ''}")
        return self.complete(
            system_prompt=system_prompt,
            user_prompt="\n".join(conversation),
            generation_config=generation_config,
        )

    def stream_messages(
        self,
        messages: list[dict[str, str | None]],
        generation_config: GenerationConfig,
    ) -> Iterator[str]:
        """Stream assistant text, falling back to one completed response chunk."""
        result = self.complete_messages(messages, generation_config)
        text = next(
            (
                str(message.get("content") or "")
                for message in reversed(result)
                if message.get("role") == "assistant"
            ),
            "",
        )
        if text:
            yield from stream_text_until_stop([text], generation_config.stop)

    def cleanup(self) -> None:  # noqa: B027 - optional hook, runners override as needed
        """
        Clean up resources (optional override).
        Default implementation does nothing.
        """
        pass
