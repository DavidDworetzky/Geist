"""
Initialize and register all available runners.
"""
import logging
from enum import Enum
from typing import Optional, Dict, Type
from agents.architectures import get_registry
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.architectures.vllm_runner import VLLMRunner

logger = logging.getLogger(__name__)

_initialized = False


class OnlineModelProviders(Enum):
    """Enum for online model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    XAI = "xai"

class OnlineModelNames(Enum):
    """Enum for online model names."""
    GPT4 = "gpt-4"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O_MICRO = "gpt-4o-micro"
    GPT4O_MICRO_2024_07_18 = "gpt-4o-micro-2024-07-18"
    GPT4O_MICRO_2024_07_18 = "gpt-4o-micro-2024-07-18"
    GROK = "grok"
    GROK_2 = "grok-2"
    GROK_2_VISION = "grok-2-vision"
    GROK_2_VISION_2 = "grok-2-vision-2"
    GROK_3 = "grok-3"
    # Anthropic models
    SONNET40 = "sonnet-40"
    SONNET45 = "sonnet-45"
    # Open Source Models
    QWEN3 = "qwen3"

class OnlineModelConfig():
    """Config for online model."""
    provider: OnlineModelProviders
    model: OnlineModelNames
    reasoning: bool = False
    streaming: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 16
    n: int = 1
    stop: Optional[str] = None
    echo: bool = False
    best_of: Optional[int] = None
    prompt_tokens: Optional[int] = None
    response_format: Optional[str] = None

OnlineModelDefaults = [
    OnlineModelConfig(
        provider=OnlineModelProviders.OPENAI,
        model=OnlineModelNames.GPT4O,
        reasoning=False,
        streaming=False,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=16,
        n=1,
        stop=None,
        echo=False,
        best_of=None,
        prompt_tokens=None,
        response_format=None
    ),
    OnlineModelConfig(
        provider=OnlineModelProviders.ANTHROPIC,
        model=OnlineModelNames.SONNET40,
        reasoning=False,
        streaming=False,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=16,
        n=1,
        stop=None,
        echo=False,
        best_of=None,
        prompt_tokens=None,
        response_format=None
    ),
    OnlineModelConfig(
        provider=OnlineModelProviders.GROQ,
        model=OnlineModelNames.SONNET45,
        reasoning=False,
        streaming=False,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=16,
        n=1,
        stop=None,
        echo=False,
        best_of=None,
        prompt_tokens=None,
        response_format=None
    ),
    OnlineModelConfig(
        provider=OnlineModelProviders.XAI,
        model=OnlineModelNames.GROK_3,
        reasoning=False,
        streaming=False,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        max_tokens=16,
        n=1,
        stop=None,
        echo=False,
        best_of=None,
        prompt_tokens=None,
        response_format=None
    )   
]


class RunnerRegistry:
    """
    Registry for managing runner classes used in agent inference.
    
    This class encapsulates the runner registration logic and provides
    a clean interface for registering, retrieving, and listing runners.
    """
    
    def __init__(self):
        """Initialize an empty runner registry."""
        self._registry: Dict[str, Type] = {}
        self._logger = logging.getLogger(__name__)
    
    def register(self, name: str, runner_class: Type) -> None:
        """
        Register a runner class with the given name.
        
        Args:
            name: Name to register the runner under
            runner_class: The runner class to register
        """
        if name in self._registry:
            self._logger.warning(f"Overriding existing runner registration for '{name}'")
        
        self._registry[name] = runner_class
        self._logger.info(f"Registered runner '{name}': {runner_class.__name__}")
    
    def get(self, name: str) -> Optional[Type]:
        """
        Get a runner class by name.
        
        Args:
            name: Name of the runner to retrieve
            
        Returns:
            The runner class if found, None otherwise
        """
        return self._registry.get(name)
    
    def list(self) -> Dict[str, Type]:
        """
        Get all registered runners.
        
        Returns:
            Dictionary of runner name to runner class mappings
        """
        return self._registry.copy()
    
    def clear(self) -> None:
        """
        Clear all registered runners. Mainly for testing purposes.
        """
        self._registry.clear()
        self._logger.info("Cleared runner registry")
    
    def __contains__(self, name: str) -> bool:
        """
        Check if a runner is registered.
        
        Args:
            name: Name of the runner to check
            
        Returns:
            True if the runner is registered, False otherwise
        """
        return name in self._registry

def get_registry() -> RunnerRegistry:
    """
    Get the global runner registry instance.
    
    Returns:
        The global RunnerRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = RunnerRegistry()
    return _registry_instance


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
