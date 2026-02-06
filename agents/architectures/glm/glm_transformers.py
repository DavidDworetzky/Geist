"""
GLM model inference using HuggingFace transformers.

Supports THUDM GLM-4 family models for local offline inference.
"""
import os
import logging
import torch
from typing import Optional, List, Dict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from agents.models.llama_completion import strings_to_message_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GLMTransformer:
    """Local GLM inference using HuggingFace transformers pipeline."""

    def __init__(
        self,
        model_id: str = "THUDM/glm-4-9b-chat",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Derive a local weights directory from the model id
        safe_name = model_id.replace("/", "_")
        self.weights_dir = os.path.join("app", "model_weights", safe_name)

        logger.info(f"Initializing GLMTransformer for {model_id}")
        self._load_model()
        logger.info("GLMTransformer initialised successfully.")

    def _load_model(self):
        """Load tokenizer and model, downloading if necessary."""
        config_path = os.path.join(self.weights_dir, "config.json")

        if os.path.exists(config_path):
            load_path = self.weights_dir
            logger.info(f"Loading GLM model from local weights: {load_path}")
        else:
            load_path = self.model_id
            logger.info(f"Loading GLM model from HuggingFace: {load_path}")
            token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                login(token=token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        logger.info(
            f"GLM model loaded. Parameters: {self.model.num_parameters():,}"
        )

    def download_model(self):
        """Download model weights to local directory for future offline use."""
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise ValueError(
                "HUGGING_FACE_HUB_TOKEN not found in environment variables"
            )
        login(token=token)

        logger.info(f"Downloading GLM model {self.model_id} to {self.weights_dir}")
        os.makedirs(self.weights_dir, exist_ok=True)
        AutoModelForCausalLM.from_pretrained(
            self.model_id, cache_dir=self.weights_dir, trust_remote_code=True
        )
        AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.weights_dir, trust_remote_code=True
        )
        logger.info("GLM model downloaded successfully.")

    def complete(
        self, system_prompt: str, user_prompt: str
    ) -> List[Dict[str, str]]:
        """
        Generate a chat completion using the GLM model.

        Args:
            system_prompt: System instructions.
            user_prompt: User message.

        Returns:
            List of message dicts compatible with LlamaCompletion.from_dict.
        """
        # Select device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal) device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")

        model = self.model.to(device)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # Use the tokenizer's chat template when available (GLM-4 supports this)
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        else:
            prompt = self._build_prompt(system_prompt, user_prompt)
            input_ids = self.tokenizer.encode(
                prompt, return_tensors="pt"
            ).to(device)

        logger.info(
            f"Generating with temperature={self.temperature}, "
            f"top_p={self.top_p}, max_new_tokens={self.max_new_tokens}"
        )

        try:
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "top_p": self.top_p,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature

            outputs = model.generate(input_ids, **gen_kwargs)

            # Decode only the newly generated tokens
            new_tokens = outputs[0][input_ids.shape[1]:]
            response_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            logger.info("GLM text generation completed successfully.")
            return strings_to_message_dict(user_prompt, response_text)

        except Exception as e:
            logger.error(f"Error during GLM text generation: {e}")
            raise

    @staticmethod
    def _build_prompt(system_prompt: str, user_prompt: str) -> str:
        """Fallback prompt template when chat template is unavailable."""
        parts = []
        if system_prompt:
            parts.append(f"[gMASK]<sop><|system|>\n{system_prompt}")
        parts.append(f"<|user|>\n{user_prompt}")
        parts.append("<|assistant|>\n")
        return "".join(parts)
