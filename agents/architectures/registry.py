"""
Initialize and register all available runners.
"""
import logging
from agents.architectures import register_runner
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.architectures.vllm_runner import VLLMRunner

logger = logging.getLogger(__name__)

def register_all_runners():
    """Register all available runners with the registry."""
    logger.info("Registering all available runners...")
    
    # Register MLX Llama runner
    register_runner("mlx_llama", MLXLlamaRunner)
    
    # Register vLLM runner (placeholder)
    register_runner("vllm", VLLMRunner)
    
    logger.info("All runners registered successfully")

# Auto-register when module is imported
register_all_runners()
