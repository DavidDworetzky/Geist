"""
Base runner abstract class for all inference backends.
"""

from abc import ABC, abstractmethod
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

    def cleanup(self) -> None:  # noqa: B027 - optional hook, runners override as needed
        """
        Clean up resources (optional override).
        Default implementation does nothing.
        """
        pass
