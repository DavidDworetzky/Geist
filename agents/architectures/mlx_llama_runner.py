"""Switchable MLX Llama runner."""

import logging
import os
from typing import Any, Protocol

from .base_runner import BaseRunner, GenerationConfig


logger = logging.getLogger(__name__)


class _MLXBackend(Protocol):
    """Shared runtime contract implemented by both MLX backends."""

    max_new_tokens: int
    temperature: float
    top_p: float

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

    @staticmethod
    def _resolve_weights_dir(device_config: dict[str, Any]) -> str | None:
        configured = device_config.get("weights_dir")
        if configured is not None:
            if not isinstance(configured, str):
                raise TypeError("weights_dir must be a string")
            return os.path.expanduser(configured)

        local_root = os.environ.get("LOCAL_WEIGHTS_DIR")
        if local_root:
            local_root = os.path.expanduser(local_root)
            if os.path.exists(os.path.join(local_root, "config.json")):
                return local_root
            return os.path.join(local_root, "llama_3_1")

        default = os.path.join("app", "model_weights", "llama_3_1")
        return default if os.path.exists(os.path.join(default, "config.json")) else None

    def load(self, model_id: str, device_config: dict[str, Any] | None = None) -> None:
        """Load the selected implementation and propagate the requested model path."""
        device_config = device_config or {}
        self.model_id = model_id
        requested = device_config.get(
            "implementation",
            os.environ.get("GEIST_MLX_IMPLEMENTATION", "manual"),
        )
        if not isinstance(requested, str):
            raise TypeError("MLX implementation must be a string")
        self.implementation = requested.strip().lower().replace("-", "_")
        if self.implementation not in self.IMPLEMENTATIONS:
            choices = ", ".join(sorted(self.IMPLEMENTATIONS))
            raise ValueError(
                f"Unknown MLX implementation '{requested}'. Expected one of: {choices}."
            )

        self.weights_dir = self._resolve_weights_dir(device_config)
        if self.implementation == "manual":
            from agents.architectures.llama.llama_mlx import LlamaMLX

            self.llama = LlamaMLX(
                max_new_tokens=16,
                model_id=model_id,
                weights_dir=self.weights_dir,
            )
        else:
            from agents.architectures.llama.mlx_lm_backend import MLXLMBackend

            self.llama = MLXLMBackend(
                max_new_tokens=16,
                model_id=model_id,
                weights_dir=self.weights_dir,
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

    def cleanup(self) -> None:
        self.llama = None
        logger.info("MLX Llama runner cleaned up")
