"""
Architecture runner registry for agent inference backends.
"""
from typing import Dict, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Registry to store runner classes
_RUNNER_REGISTRY: Dict[str, Type] = {}

def register_runner(name: str, runner_class: Type) -> None:
    """
    Register a runner class with the given name.
    
    Args:
        name: Name to register the runner under
        runner_class: The runner class to register
    """
    if name in _RUNNER_REGISTRY:
        logger.warning(f"Overriding existing runner registration for '{name}'")
    
    _RUNNER_REGISTRY[name] = runner_class
    logger.info(f"Registered runner '{name}': {runner_class.__name__}")

def get_runner(name: str) -> Optional[Type]:
    """
    Get a runner class by name.
    
    Args:
        name: Name of the runner to retrieve
        
    Returns:
        The runner class if found, None otherwise
    """
    return _RUNNER_REGISTRY.get(name)

def list_runners() -> Dict[str, Type]:
    """
    Get all registered runners.
    
    Returns:
        Dictionary of runner name to runner class mappings
    """
    return _RUNNER_REGISTRY.copy()

def clear_registry() -> None:
    """
    Clear all registered runners. Mainly for testing purposes.
    """
    global _RUNNER_REGISTRY
    _RUNNER_REGISTRY.clear()
    logger.info("Cleared runner registry")
