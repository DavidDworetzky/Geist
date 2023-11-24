from app.models.database import database
from typing import List

class AgentContext():
    def __init__(self, world_context: List[str], task_context: List[str], execution_context: List[str], subprocess_id: int = None):
        self.world_context = world_context
        self.task_context = task_context
        self.execution_context = execution_context
        # for keeping track of our subprocess id for later termination
        self.subprocess_id = subprocess_id