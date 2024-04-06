from app.models.database import database
from agents import agent_settings
from typing import List, Any
import uuid  # Added import for uuid library

class AgentContext():
    def __init__(self, settings: agent_settings, agent_id = None, world_context:List[str] = [], task_context: List[str] = [], execution_context: List[str] = [], execution_classes: List[Any] = [], subprocess_id: int = None):
        if agent_id is None:
            self.agent_id = str(uuid.uuid4())
        else:
            self.agent_id = agent_id
        self.world_context = world_context
        self.task_context = task_context
        self.execution_context = execution_context
        # for keeping track of our subprocess id for later termination
        self.subprocess_id = subprocess_id
        self.settings = settings
        self.agent_id = agent_id

    