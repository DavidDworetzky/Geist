from app.models.database import database
from agents import agent_settings
from typing import List, Any


class AgentContext():
    def __init__(self, settings: agent_settings, world_context: List[str], task_context: List[str], execution_context: List[str], execution_classes: List[Any], subprocess_id: int = None):
        if world_context is None:
            world_context = ""
        if task_context is None:
            task_context = ""
        if execution_context is None:
            execution_context = ""
        self.world_context = world_context
        self.task_context = task_context
        self.execution_context = execution_context
        # for keeping track of our subprocess id for later termination
        self.subprocess_id = subprocess_id
        self.settings = settings