"""Compatibility wrapper around the shared VLLMRunner implementation.

The Qwen3 Transformers implementation lives in vllm_runner; this module keeps
the historical "qwen3" runner key and import path working. Tests and callers
that need to patch model-loading internals should patch them on
agents.architectures.vllm_runner, where the implementation actually resolves
its symbols.
"""

from .vllm_runner import DEFAULT_QWEN3_MODEL_ID, VLLMRunner


__all__ = ["DEFAULT_QWEN3_MODEL_ID", "Qwen3Runner"]


class Qwen3Runner(VLLMRunner):
    """Backward-compatible Qwen3 runner alias built on the shared runner."""
