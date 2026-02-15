"""Compatibility wrapper around the shared VLLMRunner implementation."""
import os
import glob
import json
import torch
import transformers
import safetensors.torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import login

from . import vllm_runner as _shared
from .vllm_runner import VLLMRunner, DEFAULT_QWEN3_MODEL_ID


class Qwen3Runner(VLLMRunner):
    """Backward-compatible Qwen3 runner alias built on the shared runner."""

    @staticmethod
    def _sync_shared_symbols() -> None:
        _shared.os = os
        _shared.glob = glob
        _shared.json = json
        _shared.torch = torch
        _shared.transformers = transformers
        _shared.safetensors = safetensors
        _shared.AutoTokenizer = AutoTokenizer
        _shared.AutoModelForCausalLM = AutoModelForCausalLM
        _shared.AutoConfig = AutoConfig
        _shared.login = login

    def load(self, model_id, device_config=None):
        self._sync_shared_symbols()
        return super().load(model_id, device_config)
