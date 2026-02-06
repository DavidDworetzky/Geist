"""
Qwen 3 runner implementation using HuggingFace Transformers.

Supports Qwen 3 model variants for local inference via the
standard BaseRunner interface.
"""
from typing import Dict, Any, Optional
import logging
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from .base_runner import BaseRunner, GenerationConfig
from agents.models.llama_completion import strings_to_message_dict

logger = logging.getLogger(__name__)

# Default Qwen 3 model for local inference
DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_QWEN3_WEIGHTS_DIR = "app/model_weights/qwen3"


class Qwen3Runner(BaseRunner):
    """Runner for Qwen 3 model inference using HuggingFace Transformers."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_id = None
        self.weights_dir = None
        self.device = None

    def load(self, model_id: str, device_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load a Qwen 3 model for inference.

        Args:
            model_id: Model identifier. Can be a HuggingFace model ID
                      (e.g., "Qwen/Qwen3-8B") or a local weights directory name
                      (e.g., "Qwen3-8B").
            device_config: Optional device configuration. Supports:
                - "weights_dir": Custom path to local model weights
                - "device": Force a specific device ("cuda", "mps", "cpu")
        """
        device_config = device_config or {}
        self.model_id = model_id

        # Determine weights directory
        self.weights_dir = device_config.get(
            "weights_dir",
            os.path.join("app/model_weights", model_id.replace("/", "_"))
        )

        # Determine device
        if "device" in device_config:
            self.device = torch.device(device_config["device"])
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device for Qwen3")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) device for Qwen3")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device for Qwen3")

        # Load from local weights if available, otherwise from HuggingFace hub
        if os.path.exists(os.path.join(self.weights_dir, "config.json")):
            logger.info(f"Loading Qwen3 model from local weights: {self.weights_dir}")
            self._load_from_local()
        else:
            logger.info(f"Loading Qwen3 model from HuggingFace: {self.model_id}")
            self._load_from_hub()

        logger.info(f"Qwen3 runner loaded for model: {model_id}")

    def _load_from_local(self) -> None:
        """Load model and tokenizer from local weights directory."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.weights_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.weights_dir,
            torch_dtype=torch.float16,
        )
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded from local weights. Parameters: {self.model.num_parameters()}")

    def _load_from_hub(self) -> None:
        """Load model and tokenizer from HuggingFace hub."""
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            login(token=token)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded from HuggingFace hub. Parameters: {self.model.num_parameters()}")

    def generate(self, prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Generate text from a raw prompt.

        Args:
            prompt: Input text prompt
            generation_config: Generation parameters

        Returns:
            Dictionary containing generated messages
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.complete("", prompt, generation_config)

    def complete(self, system_prompt: str, user_prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Complete a conversation with system and user prompts using the Qwen 3 chat template.

        Args:
            system_prompt: System instructions
            user_prompt: User input
            generation_config: Generation parameters

        Returns:
            Dictionary containing completion messages in the standard format
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build messages for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Use the tokenizer's chat template for correct formatting
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple concatenation
            prompt_text = ""
            if system_prompt:
                prompt_text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            prompt_text += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            prompt_text += "<|im_start|>assistant\n"

        logger.info(
            f"Generating with temperature={generation_config.temperature}, "
            f"top_p={generation_config.top_p}, max_tokens={generation_config.max_tokens}"
        )

        try:
            pipeline = transformers.pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )

            outputs = pipeline(
                prompt_text,
                max_new_tokens=generation_config.max_tokens,
                do_sample=generation_config.temperature > 0,
                temperature=generation_config.temperature if generation_config.temperature > 0 else None,
                top_p=generation_config.top_p,
            )

            output_text = outputs[0]["generated_text"]

            # Extract the assistant response from the generated text
            # Strip the prompt from the output
            if output_text.startswith(prompt_text):
                response_text = output_text[len(prompt_text):]
            else:
                response_text = output_text

            # Clean up Qwen chat template markers from response
            for marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
                idx = response_text.find(marker)
                if idx != -1:
                    response_text = response_text[:idx]

            response_text = response_text.strip()

            logger.info("Qwen3 completion successful")
            logger.info(f"Output: {response_text}")

            return strings_to_message_dict(user_prompt, response_text)

        except Exception as e:
            logger.error(f"Error during Qwen3 completion: {str(e)}")
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen3 runner cleaned up")
