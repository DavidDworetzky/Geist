"""
Architecture runner registry for agent inference backends.
"""
from typing import Optional, Type, Dict
from .registry import RunnerRegistry, get_registry


# Legacy compatibility functions
def register_runner(name: str, runner_class: Type) -> None:
    """Register a runner class (legacy compatibility)."""
    get_registry().register(name, runner_class)


def get_runner(name: str) -> Optional[Type]:
    """Get a runner class (legacy compatibility)."""
    return get_registry().get(name)


def list_runners() -> Dict[str, Type]:
    """List all runners (legacy compatibility)."""
    return get_registry().list()


def clear_registry() -> None:
    """Clear registry (legacy compatibility)."""
    get_registry().clear()
