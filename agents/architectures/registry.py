"""
Initialize and register all available runners.
"""
import logging
from typing import Optional, Dict, Type
from agents.architectures import get_registry
from agents.architectures.mlx_llama_runner import MLXLlamaRunner
from agents.architectures.vllm_runner import VLLMRunner

logger = logging.getLogger(__name__)

_initialized = False
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
