from app.models.database import database
from agents.agent_settings import AgentSettings
from typing import List, Any
from app.models.database import agent
from app.models.database.database import Base, Session
import uuid  # Added import for uuid library

class AgentContext():
    def __init__(self, settings: AgentSettings, agent_id = None, world_context:List[str] = [], task_context: List[str] = [], execution_context: List[str] = [], execution_classes: List[Any] = [], subprocess_id: int = None):
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

    def _save(self):
        #get existing agent in database.
        session = Session
        session.query(agent).filter_by(id = self.agent_id).first()
        agent.world_context = self.world_context
        agent.task_context = self.task_context
        agent.execution_context = self.task_context
        session.commit()


    