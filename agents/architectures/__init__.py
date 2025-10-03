"""
Architecture runner registry for agent inference backends.
"""
from typing import Dict, Type, Any, Optional
import logging


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


# Global registry instance - initialized once
_registry_instance: Optional[RunnerRegistry] = None


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


# Legacy compatibility functions - delegate to the registry instance
def register_runner(name: str, runner_class: Type) -> None:
    """
    Register a runner class with the given name.
    
    DEPRECATED: Use get_registry().register() instead.
    
    Args:
        name: Name to register the runner under
        runner_class: The runner class to register
    """
    get_registry().register(name, runner_class)


def get_runner(name: str) -> Optional[Type]:
    """
    Get a runner class by name.
    
    DEPRECATED: Use get_registry().get() instead.
    
    Args:
        name: Name of the runner to retrieve
        
    Returns:
        The runner class if found, None otherwise
    """
    return get_registry().get(name)


def list_runners() -> Dict[str, Type]:
    """
    Get all registered runners.
    
    DEPRECATED: Use get_registry().list() instead.
    
    Returns:
        Dictionary of runner name to runner class mappings
    """
    return get_registry().list()


def clear_registry() -> None:
    """
    Clear all registered runners. Mainly for testing purposes.
    
    DEPRECATED: Use get_registry().clear() instead.
    """
    get_registry().clear()
