"""
HuggingFace Transformers runner for generic model support.

Supports any model available through HuggingFace's AutoModelForCausalLM,
including Kimi K2.5 (moonshotai/Kimi-K2.5) and other non-Llama architectures.
"""
from typing import Dict, Any, Optional
import logging
import os
from .base_runner import BaseRunner, GenerationConfig

logger = logging.getLogger(__name__)

# Default weights directory base path
DEFAULT_WEIGHTS_BASE = "app/model_weights"


class HuggingFaceRunner(BaseRunner):
    """Runner for HuggingFace Transformers-based inference.

    Loads models via AutoModelForCausalLM and AutoTokenizer, supporting
    any architecture available on HuggingFace Hub.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_id = None
        self.device = None

    def load(self, model_id: str, device_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load a HuggingFace model.

        Attempts to load from a local weights directory first. If not found,
        downloads from HuggingFace Hub (requires HUGGING_FACE_HUB_TOKEN).

        Args:
            model_id: HuggingFace model identifier (e.g., "moonshotai/Kimi-K2.5")
            device_config: Optional device configuration. Supported keys:
                - weights_dir: Override the local weights directory
                - torch_dtype: Override the torch dtype (default: auto)
                - device_map: Override the device map (default: auto)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        device_config = device_config or {}

        # Determine local weights directory
        safe_model_name = model_id.replace("/", "_")
        weights_dir = device_config.get(
            "weights_dir",
            os.path.join(DEFAULT_WEIGHTS_BASE, safe_model_name),
        )

        # Determine torch dtype
        torch_dtype_str = device_config.get("torch_dtype", "auto")
        if torch_dtype_str == "auto":
            torch_dtype = "auto"
        elif torch_dtype_str == "float16":
            torch_dtype = torch.float16
        elif torch_dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"

        device_map = device_config.get("device_map", "auto")

        # Determine the source to load from
        if os.path.exists(os.path.join(weights_dir, "config.json")):
            source = weights_dir
            logger.info(f"Loading model from local weights: {weights_dir}")
        else:
            source = model_id
            logger.info(f"Loading model from HuggingFace Hub: {model_id}")
            # Ensure HuggingFace token is available for gated models
            token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                from huggingface_hub import login
                login(token=token)

        logger.info(f"Loading tokenizer for {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)

        logger.info(f"Loading model {model_id} (dtype={torch_dtype_str}, device_map={device_map})")
        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) device")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")

        logger.info(f"HuggingFace runner loaded model: {model_id}")

    def generate(self, prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            generation_config: Generation parameters

        Returns:
            Dictionary with generated_text and model metadata
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.complete("", prompt, generation_config)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        """
        Complete a conversation with system and user prompts.

        Uses the model's chat template if available, falling back to a
        simple concatenation.

        Args:
            system_prompt: System instructions
            user_prompt: User input
            generation_config: Generation parameters

        Returns:
            Dictionary containing completion messages and metadata
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch
        import transformers

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": user_prompt})

        try:
            pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )

            outputs = pipeline(
                messages,
                max_new_tokens=generation_config.max_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                do_sample=generation_config.temperature > 0,
            )

            generated_messages = outputs[0]["generated_text"]
            logger.info(f"HuggingFace completion successful for model: {self.model_id}")

            # The pipeline returns the full message list including generated assistant response
            return {"messages": generated_messages}

        except Exception as e:
            logger.error(f"Error during HuggingFace completion: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        logger.info("HuggingFace runner cleaned up")
