"""
Unified runner implementation used by vLLM and Qwen3 runner entry points.

This implementation currently uses HuggingFace Transformers for inference while
keeping the vLLM runner interface stable.
"""

import glob
import json
import logging
import os
import threading
from collections.abc import Iterator
from typing import Any

import safetensors.torch
import torch
from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .base_runner import BaseRunner, GenerationConfig


logger = logging.getLogger(__name__)

DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-8B"
GENERATION_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class _StopOnEvent(StoppingCriteria):
    def __init__(self, stopped: threading.Event):
        self.stopped = stopped

    def __call__(self, _input_ids, _scores, **_kwargs) -> bool:
        return self.stopped.is_set()


class VLLMRunner(BaseRunner):
    """Runner for vLLM/Qwen3-style model inference via Transformers."""

    def __init__(self):
        self.model: Any = None
        self.tokenizer: Any = None
        self._pipeline: Any = None
        self.model_id: str | None = None
        self.weights_dir: str | None = None
        self.device: Any = None

    def load(self, model_id: str, device_config: dict[str, Any] | None = None) -> None:
        """Load a model for inference from local files or HuggingFace Hub."""
        logger.warning(
            "The 'vllm' runner is a Transformers-based shim, not a real vLLM engine "
            "(no paged attention or continuous batching); loading %s via Transformers",
            model_id,
        )
        device_config = device_config or {}
        self.model_id = model_id

        weights_dir = device_config.get(
            "weights_dir", os.path.join("app/model_weights", model_id.replace("/", "_"))
        )
        if not isinstance(weights_dir, str):
            raise TypeError("weights_dir must be a string path")
        self.weights_dir = weights_dir

        if "device" in device_config:
            self.device = torch.device(device_config["device"])
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) device")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device")

        config_path = os.path.join(self.weights_dir, "config.json")
        has_config = os.path.exists(config_path)
        safetensors_files = self._find_safetensors_files(self.weights_dir) if has_config else []
        has_hf_index = os.path.exists(
            os.path.join(self.weights_dir, "model.safetensors.index.json")
        ) or os.path.exists(os.path.join(self.weights_dir, "model.safetensors"))
        has_bin = os.path.exists(
            os.path.join(self.weights_dir, "pytorch_model.bin")
        ) or os.path.exists(os.path.join(self.weights_dir, "pytorch_model.bin.index.json"))

        if has_config and (has_hf_index or has_bin):
            logger.info(f"Loading model from HF pretrained directory: {self.weights_dir}")
            self._load_from_local()
        elif has_config and safetensors_files:
            logger.info(
                f"Loading model from safetensors files: {self.weights_dir} "
                f"({len(safetensors_files)} files)"
            )
            self._load_from_safetensors(safetensors_files)
        elif has_config:
            logger.info(f"Loading model from local directory: {self.weights_dir}")
            self._load_from_local()
        else:
            logger.info(f"Loading model from HuggingFace Hub: {self.model_id}")
            self._load_from_hub()

        logger.info(f"Runner loaded for model: {model_id}")

    @staticmethod
    def _find_safetensors_files(directory: str) -> list[str]:
        return sorted(glob.glob(os.path.join(directory, "*.safetensors")))

    def _load_from_local(self) -> None:
        weights_dir = self.weights_dir
        if weights_dir is None:
            raise RuntimeError("Runner weights directory is not configured")
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            weights_dir,
            torch_dtype=torch.float16,
        )
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded from local weights. Parameters: {self.model.num_parameters()}")

    def _load_from_safetensors(self, safetensors_files: list[str]) -> None:
        weights_dir = self.weights_dir
        if weights_dir is None:
            raise RuntimeError("Runner weights directory is not configured")
        tokenizer_files = [
            os.path.join(weights_dir, f)
            for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        ]
        if any(os.path.exists(f) for f in tokenizer_files):
            self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        else:
            logger.info("Tokenizer not found locally, loading from hub")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        config_path = os.path.join(weights_dir, "config.json")
        with open(config_path) as f:
            config_json = json.load(f)

        torch_dtype_str = str(config_json.get("torch_dtype", "")).lower()
        load_dtype = torch.bfloat16 if "bfloat16" in torch_dtype_str else torch.float16

        config = AutoConfig.from_pretrained(weights_dir)
        self.model = AutoModelForCausalLM.from_config(config)
        self.model = self.model.to(load_dtype)

        state_dict = {}
        for file_path in safetensors_files:
            logger.info(f"Loading safetensors file: {file_path}")
            tensors = safetensors.torch.load_file(file_path, device="cpu")
            state_dict.update(tensors)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading safetensors: {len(missing)} keys")
        if unexpected:
            logger.warning(f"Unexpected keys when loading safetensors: {len(unexpected)} keys")

        self.model = self.model.to(self.device)
        logger.info(f"Model loaded from safetensors. Parameters: {self.model.num_parameters()}")

    def _load_from_hub(self) -> None:
        model_id = self.model_id
        if model_id is None:
            raise RuntimeError("Runner model ID is not configured")
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            login(token=token)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        )
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded from HuggingFace hub. Parameters: {self.model.num_parameters()}")

    def _stream_messages(
        self,
        messages: list[dict[str, str | None]],
        generation_config: GenerationConfig,
    ) -> Iterator[str]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")

        normalized_messages = [
            {"role": str(message.get("role") or ""), "content": message.get("content") or ""}
            for message in messages
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = ""
            for message in normalized_messages:
                prompt_text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            prompt_text += "<|im_start|>assistant\n"

        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        stopped = threading.Event()
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": generation_config.max_tokens,
            "do_sample": generation_config.temperature > 0,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([_StopOnEvent(stopped)]),
        }
        if generation_config.temperature > 0:
            generation_kwargs["temperature"] = generation_config.temperature
            generation_kwargs["top_p"] = generation_config.top_p
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        generation_errors: list[Exception] = []

        def generate() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except Exception as error:
                generation_errors.append(error)
                streamer.end()

        generation_thread = threading.Thread(target=generate, daemon=True)
        generation_thread.start()
        try:
            yield from streamer
        finally:
            stopped.set()
            generation_thread.join(timeout=GENERATION_SHUTDOWN_TIMEOUT_SECONDS)
        if generation_thread.is_alive():
            raise RuntimeError("VLLM shim generation did not stop after stream cancellation")
        if generation_errors:
            raise generation_errors[0]

    def cleanup(self) -> None:
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Runner cleaned up")
