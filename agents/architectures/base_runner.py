"""
Base runner abstract class for all inference backends.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[str] = None

class BaseRunner(ABC):
    """Abstract base class for all inference runners."""
    
    @abstractmethod
    def load(self, model_id: str, device_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Load the model and prepare for inference.
        
        Args:
            model_id: Identifier for the model to load
            device_config: Optional device configuration (GPU, CPU, etc.)
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: Input text prompt
            generation_config: Configuration for generation parameters
            
        Returns:
            Dictionary containing generated text and metadata
        """
        pass
    
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, generation_config: GenerationConfig) -> Dict[str, Any]:
        """
        Complete a conversation with system and user prompts.
        
        Args:
            system_prompt: System instructions
            user_prompt: User input
            generation_config: Configuration for generation parameters
            
        Returns:
            Dictionary containing completion and metadata
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up resources (optional override).
        Default implementation does nothing.
        """
        pass
