"""
MLX Llama runner implementation extracted from LlamaAgent.
"""
from typing import Dict, Any, Optional
import logging
import torch
from .base_runner import BaseRunner, GenerationConfig
from agents.architectures.llama.llama_mlx import LlamaMLX
from agents.architectures.llama.llama_transformers import LlamaTransformer

logger = logging.getLogger(__name__)

class MLXLlamaRunner(BaseRunner):
    """Runner for MLX-based Llama inference."""
    
    def __init__(self):
        self.llama = None
        self.model_id = None
        
    def load(self, model_id: str, device_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load the MLX Llama model.
        
        Args:
            model_id: Identifier for the model (not used directly, uses default Llama 3.1)
            device_config: Optional device configuration
        """
        self.model_id = model_id
        
        # Determine which backend to use based on device availability
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon) device - initializing LlamaMLX")
            self.llama = LlamaMLX(max_new_tokens=16)  # Default, will be overridden by generation_config
        else:
            logger.info("Using CPU/CUDA device - initializing LlamaTransformer")
            self.llama = LlamaTransformer(max_new_tokens=16)
            
        logger.info(f"MLX Llama runner loaded for model: {model_id}")
    
    def generate(self, prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Generate text using the MLX Llama model.
        
        Args:
            prompt: Input text prompt
            generation_config: Generation parameters
            
        Returns:
            Dictionary containing generated text and metadata
        """
        if not self.llama:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Update model parameters if necessary
        if hasattr(self.llama, 'max_new_tokens'):
            self.llama.max_new_tokens = generation_config.max_tokens
        if hasattr(self.llama, 'temperature'):
            self.llama.temperature = generation_config.temperature
        if hasattr(self.llama, 'top_p'):
            self.llama.top_p = generation_config.top_p
            
        # For direct generation, we'll use the complete method with empty system prompt
        return self.complete("", prompt, generation_config)
    
    def complete(self, system_prompt: str, user_prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Complete using MLX Llama with system and user prompts.
        
        Args:
            system_prompt: System instructions
            user_prompt: User input
            generation_config: Generation parameters
            
        Returns:
            Dictionary containing completion messages and metadata
        """
        if not self.llama:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Update model parameters
        if hasattr(self.llama, 'max_new_tokens'):
            self.llama.max_new_tokens = generation_config.max_tokens
        if hasattr(self.llama, 'temperature'):
            self.llama.temperature = generation_config.temperature
        if hasattr(self.llama, 'top_p'):
            self.llama.top_p = generation_config.top_p
            
        try:
            # Call the MLX Llama complete method
            completion_result = self.llama.complete(
                system_prompt=system_prompt if system_prompt else "You are a helpful assistant.",
                user_prompt=user_prompt
            )
            
            logger.info(f"MLX Llama completion successful: {completion_result}")
            return completion_result
            
        except Exception as e:
            logger.error(f"Error during MLX Llama completion: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up MLX Llama resources."""
        if self.llama:
            # MLX models should clean up automatically, but we can explicitly clear
            self.llama = None
            logger.info("MLX Llama runner cleaned up")
