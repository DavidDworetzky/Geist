"""
GLM runner implementation for local offline inference.
"""
from typing import Dict, Any, Optional
import logging
from .base_runner import BaseRunner, GenerationConfig
from agents.architectures.glm.glm_transformers import GLMTransformer

logger = logging.getLogger(__name__)


class GLMRunner(BaseRunner):
    """Runner for local GLM model inference via HuggingFace transformers."""

    def __init__(self):
        self.glm = None
        self.model_id = None

    def load(self, model_id: str, device_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load a GLM model for local inference.

        Args:
            model_id: HuggingFace model identifier (e.g. "THUDM/glm-4-9b-chat").
            device_config: Optional device configuration (unused currently;
                           device selection happens automatically in GLMTransformer).
        """
        self.model_id = model_id
        logger.info(f"Loading GLM model: {model_id}")
        self.glm = GLMTransformer(
            model_id=model_id,
            max_new_tokens=16,  # default; overridden per-request via GenerationConfig
        )
        logger.info(f"GLM runner loaded for model: {model_id}")

    def generate(self, prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Generate text from a raw prompt.

        Args:
            prompt: Input text prompt.
            generation_config: Generation parameters.

        Returns:
            Dictionary containing generated text and metadata.
        """
        if not self.glm:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Delegate to complete with an empty system prompt
        return self.complete("", prompt, generation_config)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        """
        Complete a conversation using the GLM model.

        Args:
            system_prompt: System instructions.
            user_prompt: User input.
            generation_config: Generation parameters.

        Returns:
            Dictionary containing completion messages compatible with
            LlamaCompletion.from_dict.
        """
        if not self.glm:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Apply per-request generation parameters
        self.glm.max_new_tokens = generation_config.max_tokens
        if hasattr(self.glm, "temperature"):
            self.glm.temperature = generation_config.temperature
        if hasattr(self.glm, "top_p"):
            self.glm.top_p = generation_config.top_p

        try:
            result = self.glm.complete(
                system_prompt=system_prompt if system_prompt else "You are a helpful assistant.",
                user_prompt=user_prompt,
            )
            logger.info(f"GLM completion successful: {result}")
            return result
        except Exception as e:
            logger.error(f"Error during GLM completion: {e}")
            raise

    def cleanup(self) -> None:
        """Release GLM model resources."""
        if self.glm:
            self.glm = None
            logger.info("GLM runner cleaned up")
