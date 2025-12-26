"""
Initialize and register all available runners.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Type
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.architectures.vllm_runner import VLLMRunner

logger = logging.getLogger(__name__)

_initialized = False
_registry_instance = None


class OnlineModelProviders(Enum):
    """Enum for online model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    XAI = "xai"
    HUGGINGFACE = "huggingface"
    OFFLINE = "offline"

class OnlineModelNames(Enum):
    """Enum for online model names."""
    # OpenAI models
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O_MICRO = "gpt-4o-micro"
    GPT4O_MICRO_2024_07_18 = "gpt-4o-micro-2024-07-18"
    GPT35_TURBO = "gpt-3.5-turbo"
    O1 = "o1"
    O1_MINI = "o1-mini"
    O3_MINI = "o3-mini"
    # XAI models
    GROK = "grok"
    GROK_2 = "grok-2"
    GROK_2_VISION = "grok-2-vision"
    GROK_2_VISION_2 = "grok-2-vision-2"
    GROK_3 = "grok-3"
    # Anthropic models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_35_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_35_HAIKU = "claude-3-5-haiku-20241022"
    SONNET40 = "sonnet-40"
    SONNET45 = "sonnet-45"
    # Groq models
    LLAMA_33_70B = "llama-3.3-70b-versatile"
    LLAMA_31_8B = "llama-3.1-8b-instant"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    # Open Source / Huggingface Models
    QWEN3 = "qwen3"
    QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
    META_LLAMA_31_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    MIXTRAL_8X7B_HF = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # Offline / Local models
    META_LLAMA_31_8B_LOCAL = "Meta-Llama-3.1-8B-Instruct"
    META_LLAMA_31_8B_BASE_LOCAL = "Meta-Llama-3.1-8B"
    META_LLAMA_3_8B_LOCAL = "Meta-Llama-3-8B-Instruct"

@dataclass
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
        provider=OnlineModelProviders.ANTHROPIC,
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


# Convenience functions for test compatibility
def register_runner(name: str, runner_class: Type) -> None:
    """
    Register a runner class with the given name.

    Args:
        name: Name to register the runner under
        runner_class: The runner class to register
    """
    registry = get_registry()
    registry.register(name, runner_class)


def get_runner(name: str) -> Optional[Type]:
    """
    Get a runner class by name.

    Args:
        name: Name of the runner to retrieve

    Returns:
        The runner class if found, None otherwise
    """
    registry = get_registry()
    return registry.get(name)


def clear_registry() -> None:
    """
    Clear all registered runners. Mainly for testing purposes.
    """
    global _initialized
    registry = get_registry()
    registry.clear()
    _initialized = False


# ============================================================================
# Model Registry - Available models with metadata
# ============================================================================

from datetime import datetime
from typing import List

@dataclass
class ModelInfo:
    """Information about an available model with full metadata."""
    id: str                              # Model identifier (e.g., "gpt-4")
    name: str                            # Display name (e.g., "GPT-4")
    provider: OnlineModelProviders       # Provider enum
    context_window: Optional[int] = None # Max context tokens
    max_output_tokens: Optional[int] = None  # Max output tokens
    supports_vision: bool = False        # Multimodal support
    supports_function_calling: bool = False
    supports_streaming: bool = True
    recommended: bool = False            # Highlight recommended models
    family: Optional[str] = None         # Model family (e.g., "gpt-4", "claude-3")

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider.value,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_vision": self.supports_vision,
            "supports_function_calling": self.supports_function_calling,
            "supports_streaming": self.supports_streaming,
            "recommended": self.recommended,
            "family": self.family
        }


# Static registry - always available as fallback
STATIC_MODELS: Dict[OnlineModelProviders, List[ModelInfo]] = {
    OnlineModelProviders.OPENAI: [
        ModelInfo(
            id="gpt-4",
            name="GPT-4",
            provider=OnlineModelProviders.OPENAI,
            context_window=8192,
            max_output_tokens=4096,
            supports_vision=False,
            supports_function_calling=True,
            recommended=False,
            family="gpt-4"
        ),
        ModelInfo(
            id="gpt-4-turbo",
            name="GPT-4 Turbo",
            provider=OnlineModelProviders.OPENAI,
            context_window=128000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="gpt-4"
        ),
        ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            provider=OnlineModelProviders.OPENAI,
            context_window=128000,
            max_output_tokens=16384,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="gpt-4o"
        ),
        ModelInfo(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            provider=OnlineModelProviders.OPENAI,
            context_window=128000,
            max_output_tokens=16384,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="gpt-4o"
        ),
        ModelInfo(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider=OnlineModelProviders.OPENAI,
            context_window=16385,
            max_output_tokens=4096,
            supports_vision=False,
            supports_function_calling=True,
            recommended=False,
            family="gpt-3.5"
        ),
        ModelInfo(
            id="o1",
            name="O1",
            provider=OnlineModelProviders.OPENAI,
            context_window=200000,
            max_output_tokens=100000,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="o1"
        ),
        ModelInfo(
            id="o1-mini",
            name="O1 Mini",
            provider=OnlineModelProviders.OPENAI,
            context_window=128000,
            max_output_tokens=65536,
            supports_vision=False,
            supports_function_calling=True,
            recommended=True,
            family="o1"
        ),
        ModelInfo(
            id="o3-mini",
            name="O3 Mini",
            provider=OnlineModelProviders.OPENAI,
            context_window=200000,
            max_output_tokens=100000,
            supports_vision=False,
            supports_function_calling=True,
            recommended=True,
            family="o3"
        ),
    ],
    OnlineModelProviders.ANTHROPIC: [
        ModelInfo(
            id="claude-3-opus-20240229",
            name="Claude 3 Opus",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="claude-3"
        ),
        ModelInfo(
            id="claude-3-sonnet-20240229",
            name="Claude 3 Sonnet",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="claude-3"
        ),
        ModelInfo(
            id="claude-3-haiku-20240307",
            name="Claude 3 Haiku",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="claude-3"
        ),
        ModelInfo(
            id="claude-3-5-sonnet-20241022",
            name="Claude 3.5 Sonnet",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=8192,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="claude-3.5"
        ),
        ModelInfo(
            id="claude-3-5-haiku-20241022",
            name="Claude 3.5 Haiku",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=8192,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="claude-3.5"
        ),
    ],
    OnlineModelProviders.XAI: [
        ModelInfo(
            id="grok-2",
            name="Grok 2",
            provider=OnlineModelProviders.XAI,
            context_window=131072,
            max_output_tokens=32768,
            supports_vision=False,
            supports_function_calling=True,
            recommended=True,
            family="grok"
        ),
        ModelInfo(
            id="grok-2-vision",
            name="Grok 2 Vision",
            provider=OnlineModelProviders.XAI,
            context_window=32768,
            max_output_tokens=8192,
            supports_vision=True,
            supports_function_calling=True,
            recommended=True,
            family="grok"
        ),
        ModelInfo(
            id="grok-3",
            name="Grok 3",
            provider=OnlineModelProviders.XAI,
            context_window=131072,
            max_output_tokens=32768,
            supports_vision=False,
            supports_function_calling=True,
            recommended=True,
            family="grok"
        ),
    ],
    OnlineModelProviders.GROQ: [
        ModelInfo(
            id="llama-3.3-70b-versatile",
            name="Llama 3.3 70B Versatile",
            provider=OnlineModelProviders.GROQ,
            context_window=128000,
            max_output_tokens=32768,
            supports_vision=False,
            supports_function_calling=True,
            recommended=True,
            family="llama-3"
        ),
        ModelInfo(
            id="llama-3.1-8b-instant",
            name="Llama 3.1 8B Instant",
            provider=OnlineModelProviders.GROQ,
            context_window=128000,
            max_output_tokens=8192,
            supports_vision=False,
            supports_function_calling=True,
            recommended=True,
            family="llama-3"
        ),
        ModelInfo(
            id="mixtral-8x7b-32768",
            name="Mixtral 8x7B",
            provider=OnlineModelProviders.GROQ,
            context_window=32768,
            max_output_tokens=8192,
            supports_vision=False,
            supports_function_calling=True,
            recommended=False,
            family="mixtral"
        ),
    ],
    OnlineModelProviders.HUGGINGFACE: [
        ModelInfo(
            id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            name="Meta Llama 3.1 8B Instruct",
            provider=OnlineModelProviders.HUGGINGFACE,
            context_window=131072,
            max_output_tokens=8192,
            supports_vision=False,
            supports_function_calling=False,
            recommended=True,
            family="llama-3"
        ),
        ModelInfo(
            id="Qwen/Qwen2.5-72B-Instruct",
            name="Qwen 2.5 72B Instruct",
            provider=OnlineModelProviders.HUGGINGFACE,
            context_window=131072,
            max_output_tokens=8192,
            supports_vision=False,
            supports_function_calling=True,
            recommended=True,
            family="qwen"
        ),
        ModelInfo(
            id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            name="Mixtral 8x7B Instruct",
            provider=OnlineModelProviders.HUGGINGFACE,
            context_window=32768,
            max_output_tokens=8192,
            supports_vision=False,
            supports_function_calling=False,
            recommended=True,
            family="mixtral"
        ),
    ],
    OnlineModelProviders.OFFLINE: [
        ModelInfo(
            id="Meta-Llama-3.1-8B-Instruct",
            name="Meta Llama 3.1 8B Instruct (Local)",
            provider=OnlineModelProviders.OFFLINE,
            context_window=131072,
            max_output_tokens=8192,
            supports_vision=False,
            supports_function_calling=False,
            recommended=True,
            family="llama-3"
        ),
        ModelInfo(
            id="Meta-Llama-3.1-8B",
            name="Meta Llama 3.1 8B (Local)",
            provider=OnlineModelProviders.OFFLINE,
            context_window=131072,
            max_output_tokens=8192,
            supports_vision=False,
            supports_function_calling=False,
            recommended=False,
            family="llama-3"
        ),
        ModelInfo(
            id="Meta-Llama-3-8B-Instruct",
            name="Meta Llama 3 8B Instruct (Local)",
            provider=OnlineModelProviders.OFFLINE,
            context_window=8192,
            max_output_tokens=4096,
            supports_vision=False,
            supports_function_calling=False,
            recommended=False,
            family="llama-3"
        ),
    ],
}


# Dynamic registry - populated by sync script
# Auto-generated by scripts/sync_models.py
# Do not edit manually

DISCOVERED_MODELS: Dict[OnlineModelProviders, List[ModelInfo]] = {
    OnlineModelProviders.OPENAI: [
        ModelInfo(
            id="gpt-4-0613",
            name="GPT 4 0613",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4",
            name="GPT-4",
            provider=OnlineModelProviders.OPENAI,
            context_window=8192,
            max_output_tokens=4096,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
            family="gpt-4",
        ),
        ModelInfo(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider=OnlineModelProviders.OPENAI,
            context_window=16385,
            max_output_tokens=4096,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
            family="gpt-3.5",
        ),
        ModelInfo(
            id="gpt-3.5-turbo-instruct-0914",
            name="GPT 3.5 Turbo Instruct 0914",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4-1106-preview",
            name="GPT 4 1106 Preview",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-3.5-turbo-1106",
            name="GPT 3.5 Turbo 1106",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4-0125-preview",
            name="GPT 4 0125 Preview",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4-turbo-preview",
            name="GPT 4 Turbo Preview",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-3.5-turbo-0125",
            name="GPT 3.5 Turbo 0125",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4-turbo",
            name="GPT-4 Turbo",
            provider=OnlineModelProviders.OPENAI,
            context_window=128000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="gpt-4",
        ),
        ModelInfo(
            id="gpt-4-turbo-2024-04-09",
            name="GPT 4 Turbo 2024 04 09",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            provider=OnlineModelProviders.OPENAI,
            context_window=128000,
            max_output_tokens=16384,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="gpt-4o",
        ),
        ModelInfo(
            id="gpt-4o-2024-05-13",
            name="GPT 4o 2024 05 13",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-mini-2024-07-18",
            name="GPT 4o Mini 2024 07 18",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            provider=OnlineModelProviders.OPENAI,
            context_window=128000,
            max_output_tokens=16384,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="gpt-4o",
        ),
        ModelInfo(
            id="gpt-4o-2024-08-06",
            name="GPT 4o 2024 08 06",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-audio-preview",
            name="GPT 4o Audio Preview",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-realtime-preview",
            name="GPT 4o Realtime Preview",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-realtime-preview-2024-12-17",
            name="GPT 4o Realtime Preview 2024 12 17",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-audio-preview-2024-12-17",
            name="GPT 4o Audio Preview 2024 12 17",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-mini-realtime-preview-2024-12-17",
            name="GPT 4o Mini Realtime Preview 2024 12 17",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-mini-audio-preview-2024-12-17",
            name="GPT 4o Mini Audio Preview 2024 12 17",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="o1-2024-12-17",
            name="O1 2024 12 17",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="o1",
            name="O1",
            provider=OnlineModelProviders.OPENAI,
            context_window=200000,
            max_output_tokens=100000,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="o1",
        ),
        ModelInfo(
            id="gpt-4o-mini-realtime-preview",
            name="GPT 4o Mini Realtime Preview",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-mini-audio-preview",
            name="GPT 4o Mini Audio Preview",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="o3-mini",
            name="O3 Mini",
            provider=OnlineModelProviders.OPENAI,
            context_window=200000,
            max_output_tokens=100000,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="o3",
        ),
        ModelInfo(
            id="o3-mini-2025-01-31",
            name="O3 Mini 2025 01 31",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-2024-11-20",
            name="GPT 4o 2024 11 20",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-transcribe",
            name="GPT 4o Transcribe",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-mini-transcribe",
            name="GPT 4o Mini Transcribe",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="o1-pro-2025-03-19",
            name="O1 Pro 2025 03 19",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="o1-pro",
            name="O1 Pro",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="o3-2025-04-16",
            name="O3 2025 04 16",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="o3",
            name="O3",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4.1-2025-04-14",
            name="GPT 4.1 2025 04 14",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4.1",
            name="GPT 4.1",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4.1-mini-2025-04-14",
            name="GPT 4.1 Mini 2025 04 14",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4.1-mini",
            name="GPT 4.1 Mini",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4.1-nano-2025-04-14",
            name="GPT 4.1 Nano 2025 04 14",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4.1-nano",
            name="GPT 4.1 Nano",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-realtime-preview-2025-06-03",
            name="GPT 4o Realtime Preview 2025 06 03",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-audio-preview-2025-06-03",
            name="GPT 4o Audio Preview 2025 06 03",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-transcribe-diarize",
            name="GPT 4o Transcribe Diarize",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-mini-transcribe-2025-12-15",
            name="GPT 4o Mini Transcribe 2025 12 15",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-4o-mini-transcribe-2025-03-20",
            name="GPT 4o Mini Transcribe 2025 03 20",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
        ModelInfo(
            id="gpt-3.5-turbo-16k",
            name="GPT 3.5 Turbo 16k",
            provider=OnlineModelProviders.OPENAI,
            supports_vision=False,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=False,
        ),
    ],
    OnlineModelProviders.ANTHROPIC: [
        ModelInfo(
            id="claude-opus-4-5-20251101",
            name="Claude Opus 4.5",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="claude",
        ),
        ModelInfo(
            id="claude-haiku-4-5-20251001",
            name="Claude Haiku 4.5",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="claude",
        ),
        ModelInfo(
            id="claude-sonnet-4-5-20250929",
            name="Claude Sonnet 4.5",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="claude",
        ),
        ModelInfo(
            id="claude-opus-4-1-20250805",
            name="Claude Opus 4.1",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="claude",
        ),
        ModelInfo(
            id="claude-opus-4-20250514",
            name="Claude Opus 4",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="claude",
        ),
        ModelInfo(
            id="claude-sonnet-4-20250514",
            name="Claude Sonnet 4",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="claude",
        ),
        ModelInfo(
            id="claude-3-5-haiku-20241022",
            name="Claude Haiku 3.5",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=8192,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="claude-3.5",
        ),
        ModelInfo(
            id="claude-3-haiku-20240307",
            name="Claude Haiku 3",
            provider=OnlineModelProviders.ANTHROPIC,
            context_window=200000,
            max_output_tokens=4096,
            supports_vision=True,
            supports_function_calling=True,
            supports_streaming=True,
            recommended=True,
            family="claude-3",
        ),
    ],
    OnlineModelProviders.HUGGINGFACE: [
        ModelInfo(
            id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            name="Meta Llama 3.1 8B Instruct",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="meta-llama",
        ),
        ModelInfo(
            id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            name="Meta Llama 3.1 70B Instruct",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="meta-llama",
        ),
        ModelInfo(
            id="meta-llama/Llama-3.3-70B-Instruct",
            name="Llama 3.3 70B Instruct",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="meta-llama",
        ),
        ModelInfo(
            id="Qwen/Qwen2.5-72B-Instruct",
            name="Qwen2.5 72B Instruct",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="Qwen",
        ),
        ModelInfo(
            id="Qwen/Qwen2.5-7B-Instruct",
            name="Qwen2.5 7B Instruct",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="Qwen",
        ),
        ModelInfo(
            id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            name="Mixtral 8X7B Instruct V0.1",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="mistralai",
        ),
        ModelInfo(
            id="mistralai/Mistral-7B-Instruct-v0.3",
            name="Mistral 7B Instruct V0.3",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="mistralai",
        ),
        ModelInfo(
            id="deepseek-ai/DeepSeek-V2.5",
            name="Deepseek V2.5",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="deepseek-ai",
        ),
        ModelInfo(
            id="THUDM/glm-4-9b-chat",
            name="Glm 4 9B Chat",
            provider=OnlineModelProviders.HUGGINGFACE,
            supports_vision=False,
            supports_function_calling=False,
            supports_streaming=True,
            recommended=True,
            family="THUDM",
        ),
    ],
}

# Timestamp of last model sync
_last_model_sync: Optional[datetime] = None


def get_models_for_provider(provider: OnlineModelProviders) -> List[ModelInfo]:
    """
    Get all available models for a provider, preferring discovered models.

    Args:
        provider: The provider to get models for

    Returns:
        List of ModelInfo objects for the provider
    """
    # Prefer discovered models if available
    if provider in DISCOVERED_MODELS and DISCOVERED_MODELS[provider]:
        return DISCOVERED_MODELS[provider]
    else:
        return STATIC_MODELS.get(provider, [])


def get_all_models() -> Dict[OnlineModelProviders, List[ModelInfo]]:
    """
    Get all available models grouped by provider.

    Returns:
        Dictionary mapping Provider to list of ModelInfo
    """
    result = {}
    for provider in OnlineModelProviders:
        models = get_models_for_provider(provider)
        if models:  # Only include providers with models
            result[provider] = models
    return result


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """
    Find a model by its ID across all providers.

    Args:
        model_id: The model identifier to find

    Returns:
        ModelInfo if found, None otherwise
    """
    for provider in OnlineModelProviders:
        models = get_models_for_provider(provider)
        for model in models:
            if model.id == model_id:
                return model
    return None


def update_discovered_models(provider: OnlineModelProviders, models: List[ModelInfo]) -> None:
    """
    Update the discovered models for a provider.

    Args:
        provider: The provider to update
        models: List of discovered models
    """
    global _last_model_sync
    DISCOVERED_MODELS[provider] = models
    _last_model_sync = datetime.utcnow()
    logger.info(f"Updated discovered models for {provider.value}: {len(models)} models")


def get_last_model_sync_time() -> Optional[datetime]:
    """Get the timestamp of the last model sync."""
    return _last_model_sync


def provider_from_string(provider_str: str) -> Optional[OnlineModelProviders]:
    """
    Convert a provider string to OnlineModelProviders enum.

    Args:
        provider_str: Provider string (e.g., "openai", "anthropic")

    Returns:
        OnlineModelProviders enum value if valid, None otherwise
    """
    try:
        return OnlineModelProviders(provider_str.lower())
    except ValueError:
        return None
