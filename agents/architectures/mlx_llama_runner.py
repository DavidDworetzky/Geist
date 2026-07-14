"""Switchable MLX Llama runner."""

import logging
import os
from typing import Any

from .base_runner import BaseRunner, GenerationConfig


logger = logging.getLogger(__name__)


class MLXLlamaRunner(BaseRunner):
    """Run Llama through Geist's manual MLX code or the mlx-lm adapter."""

    IMPLEMENTATIONS = {"manual", "mlx_lm"}

    def __init__(self):
        self.llama = None
        self.model_id = None
        self.weights_dir = None
        self.implementation = None

    @staticmethod
    def _resolve_weights_dir(device_config: dict[str, Any]) -> str | None:
        configured = device_config.get("weights_dir")
        if configured:
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
        self.implementation = requested.strip().lower().replace("-", "_")
        if self.implementation not in self.IMPLEMENTATIONS:
            choices = ", ".join(sorted(self.IMPLEMENTATIONS))
            raise ValueError(
                f"Unknown MLX implementation '{requested}'. Expected one of: {choices}."
            )

        self.weights_dir = self._resolve_weights_dir(device_config)
        backend_args = {
            "max_new_tokens": 16,
            "model_id": model_id,
            "weights_dir": self.weights_dir,
        }
        if self.implementation == "manual":
            from agents.architectures.llama.llama_mlx import LlamaMLX

            self.llama = LlamaMLX(**backend_args)
        else:
            from agents.architectures.llama.mlx_lm_backend import MLXLMBackend

            self.llama = MLXLMBackend(**backend_args)

        logger.info(
            "MLX Llama runner loaded implementation=%s model=%s weights=%s",
            self.implementation,
            model_id,
            self.weights_dir or "Hugging Face",
        )

    def _apply_generation_config(self, generation_config: GenerationConfig) -> None:
        if not self.llama:
            raise RuntimeError("Model not loaded. Call load() first.")
        self.llama.max_new_tokens = generation_config.max_tokens
        self.llama.temperature = generation_config.temperature
        self.llama.top_p = generation_config.top_p

    def generate(self, prompt: str, generation_config: GenerationConfig) -> dict[str, Any]:
        return self.complete("", prompt, generation_config)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        self._apply_generation_config(generation_config)
        return self.llama.complete(
            system_prompt=system_prompt or "You are a helpful assistant.",
            user_prompt=user_prompt,
        )

    def cleanup(self) -> None:
        self.llama = None
        logger.info("MLX Llama runner cleaned up")
