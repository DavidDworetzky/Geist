"""
Initialize and register all available runners.
"""
import logging
from typing import Optional
from agents.architectures import RunnerRegistry, get_registry
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.architectures.vllm_runner import VLLMRunner

logger = logging.getLogger(__name__)

_initialized = False


def register_all_runners(registry: Optional[RunnerRegistry] = None) -> None:
    """
    Register all available runners with the registry.
    
    Args:
        registry: Optional RunnerRegistry instance. If None, uses the global registry.
    """
    global _initialized
    
    if registry is None:
        registry = get_registry()
    
    logger.info("Registering all available runners...")
    
    # Register MLX Llama runner
    registry.register("mlx_llama", MLXLlamaRunner)
    
    # Register vLLM runner (placeholder)
    registry.register("vllm", VLLMRunner)
    
    _initialized = True
    logger.info("All runners registered successfully")


def ensure_runners_registered() -> None:
    """
    Ensure that runners are registered. Call this before using any runners.
    This is idempotent - it will only register once.
    """
    global _initialized
    if not _initialized:
        register_all_runners()
