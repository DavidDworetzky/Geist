"""Generic, performance-aware Hugging Face causal language-model runner."""
from __future__ import annotations

import gc
import importlib.util
import logging
import os
import re
from contextlib import suppress
from importlib import metadata
from typing import Any, cast

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from agents.architectures.base_runner import BaseRunner, GenerationConfig
from agents.model_catalog import infer_model_spec
from agents.models.llama_completion import strings_to_message_dict


logger = logging.getLogger(__name__)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


class TransformersRunner(BaseRunner):
    """Run standard Hugging Face causal LMs without family-specific code."""

    def __init__(self):
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.config: Any | None = None
        self.model_id: str | None = None
        self.source: str | None = None
        self.device: torch.device | None = None
        self.model_spec = None
        self.trust_remote_code = False

    def load(self, model_id: str, device_config: dict[str, Any] | None = None) -> None:
        device_config = dict(device_config or {})
        allow_server_backed = bool(device_config.pop("allow_server_backed", False))
        self.model_id = model_id
        self.model_spec = infer_model_spec(model_id)
        if self.model_spec and not self.model_spec.local and not allow_server_backed:
            raise ValueError(
                f"{model_id} is cataloged as server-backed. Configure it as an online "
                f"model using provider '{self.model_spec.provider}'."
            )

        self._check_transformers_version()
        self.trust_remote_code = bool(device_config.pop("trust_remote_code", False))
        if self.model_spec and self.model_spec.requires_remote_code and not self.trust_remote_code:
            raise ValueError(
                f"{model_id} requires remote model code. Set trust_remote_code=true only "
                "after reviewing the repository, or use an OpenAI-compatible server."
            )
        if self.model_spec:
            missing_optional = [
                dependency
                for dependency in self.model_spec.optional_dependencies
                if importlib.util.find_spec(dependency) is None
            ]
            if missing_optional:
                logger.warning(
                    "%s is missing optional acceleration packages: %s",
                    model_id,
                    ", ".join(missing_optional),
                )

        weights_dir = device_config.pop("weights_dir", None)
        default_dir = os.path.join("app", "model_weights", model_id.replace("/", "_"))
        self.source = weights_dir or (default_dir if os.path.isdir(default_dir) else model_id)
        explicit_device = device_config.pop("device", None)
        self.device = self._select_device(explicit_device)
        compile_model = bool(device_config.pop("compile", False))
        if self.device.type == "cuda" and device_config.pop("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        common_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if token:
            common_kwargs["token"] = token
        for option in ("revision", "cache_dir", "local_files_only", "subfolder"):
            if option in device_config:
                common_kwargs[option] = device_config.pop(option)

        self.config = AutoConfig.from_pretrained(self.source, **common_kwargs)
        architectures = getattr(self.config, "architectures", None) or []
        if any("ConditionalGeneration" in name or "Multimodal" in name for name in architectures):
            raise ValueError(
                f"{model_id} is multimodal and needs a processor/conditional-generation runner; "
                "it cannot be loaded by the text-only Transformers runner."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.source, **common_kwargs)
        load_kwargs: dict[str, Any] = dict(common_kwargs)
        load_kwargs["torch_dtype"] = self._select_dtype(device_config.pop("dtype", None))
        if self.device.type == "mps" and "attn_implementation" not in device_config:
            # Torch 2.6 SDPA can abort the process for otherwise valid causal-LM
            # shapes on Apple MPS. Eager attention is slower but stable, and an
            # explicit device option can still opt into a newer implementation.
            load_kwargs["attn_implementation"] = "eager"

        has_accelerate = importlib.util.find_spec("accelerate") is not None
        if has_accelerate:
            load_kwargs["low_cpu_mem_usage"] = True
            requested_device_map = device_config.pop("device_map", None)
            if requested_device_map is not None:
                load_kwargs["device_map"] = requested_device_map
            elif self.device.type == "cuda" and explicit_device is None:
                load_kwargs["device_map"] = "auto"

        for option in (
            "attn_implementation", "quantization_config", "load_in_4bit",
            "load_in_8bit", "max_memory", "offload_folder",
            "offload_state_dict", "use_safetensors",
        ):
            if option in device_config:
                load_kwargs[option] = device_config.pop(option)
        if device_config:
            logger.warning("Ignoring unsupported Transformers device options: %s", sorted(device_config))

        logger.info(
            "Loading %s with Transformers on %s (%s)",
            model_id, self.device, load_kwargs["torch_dtype"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.source, **load_kwargs)
        if not getattr(self.model, "hf_device_map", None):
            self.model = cast(Any, self.model).to(self.device)
        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        if compile_model and getattr(self.model, "hf_device_map", None):
            raise ValueError("torch.compile is not supported with a sharded device_map")
        if compile_model:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch build")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _check_transformers_version(self) -> None:
        required = self.model_spec.min_transformers_version if self.model_spec else None
        if not required:
            return
        installed = metadata.version("transformers")
        if _version_tuple(installed) < _version_tuple(required):
            raise RuntimeError(
                f"{self.model_id} requires transformers>={required}; installed version is {installed}. "
                "Upgrade the environment or use an OpenAI-compatible inference server."
            )

    @staticmethod
    def _select_device(explicit: str | None) -> torch.device:
        if explicit:
            return torch.device(explicit)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _select_dtype(self, explicit: Any | None):
        if explicit is not None:
            if isinstance(explicit, torch.dtype):
                return explicit
            value = str(explicit).replace("torch.", "").lower()
            dtypes = {
                "auto": "auto", "float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            }
            if value not in dtypes:
                raise ValueError(f"Unsupported dtype: {explicit}")
            return dtypes[value]
        if self.device is None:
            raise RuntimeError("Device not selected. Call load() first.")
        if self.device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self.device.type == "mps":
            return torch.float16
        return torch.float32

    def generate(
        self, prompt: str, generation_config: GenerationConfig
    ) -> list[dict[str, str]]:
        return self.complete("", prompt, generation_config)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        generation_config: GenerationConfig,
    ) -> list[dict[str, str]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"{self.model_id} does not provide a tokenizer chat template. "
                "Use an instruction-tuned checkpoint with a published chat template."
            )

        try:
            inputs = self._apply_chat_template(messages)
        except Exception:
            if not system_prompt:
                raise
            # A few otherwise standard instruction checkpoints do not define a
            # system role. Preserve the instruction without family-specific
            # branching by placing it ahead of the user content.
            logger.info(
                "%s chat template rejected a system role; folding it into the user turn",
                self.model_id,
            )
            inputs = self._apply_chat_template([
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ])
        inputs = dict(inputs) if hasattr(inputs, "items") else {"input_ids": inputs}
        target_device = getattr(self.model, "device", self.device)
        inputs = {name: value.to(target_device) for name, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        max_context = getattr(self.config, "max_position_embeddings", None)
        max_new_tokens = generation_config.max_tokens
        if isinstance(max_context, int) and max_context > 0:
            available_tokens = max_context - input_length
            if available_tokens <= 0:
                raise ValueError(
                    f"Prompt uses {input_length} tokens, exceeding the model context "
                    f"window of {max_context}."
                )
            if max_new_tokens > available_tokens:
                logger.warning(
                    "Clamping max_new_tokens from %s to %s for model context limit",
                    max_new_tokens, available_tokens,
                )
                max_new_tokens = available_tokens

        do_sample = generation_config.temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = generation_config.temperature
            generation_kwargs["top_p"] = generation_config.top_p

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        response = self.tokenizer.decode(
            generated[0][input_length:], skip_special_tokens=True
        ).strip()
        if generation_config.stop:
            stop_sequences = (
                [generation_config.stop]
                if isinstance(generation_config.stop, str)
                else generation_config.stop
            )
            stop_positions = [
                position
                for stop_sequence in stop_sequences
                if stop_sequence
                if (position := response.find(stop_sequence)) >= 0
            ]
            if stop_positions:
                response = response[: min(stop_positions)].rstrip()
        return strings_to_message_dict(user_prompt, response)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> Any:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def cleanup(self) -> None:
        self.model = None
        self.tokenizer = None
        self.config = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            with suppress(RuntimeError):
                torch.mps.empty_cache()
