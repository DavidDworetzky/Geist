"""
Architecture runner registry for agent inference backends.
"""
from typing import Optional
from registry import RunnerRegistry



# Global registry instance - initialized once
_registry_instance: Optional[RunnerRegistry] = None
