from app.models.database import database
from agents.agent_settings import AgentSettings
from typing import List, Any, Dict
from app.models.database import agent
from app.models.database.database import Base, Session
import uuid  # Added import for uuid library
from adapters.adapter_registry import find_adapter_classes, init_adapter_class

class AgentContext():
    def __init__(self, settings: AgentSettings, agent_id = None, world_context:List[str] = [], task_context: List[str] = [], execution_context: List[str] = [], function_log: List[str] = [], execution_classes: List[Any] = [], subprocess_id: int = None, envs : Dict[str, str] = {}, include_world_processing = False):
        if agent_id is None:
            self.agent_id = str(uuid.uuid4())
        else:
            self.agent_id = agent_id
        self.world_context = world_context
        self.task_context = task_context
        self.execution_context = execution_context
        #function log and execution classes
        self.function_log = function_log
        self.execution_classes = execution_classes
        if not self.execution_classes:
            #if self.execution_classes are not defined, pull from adapter_registry
            self.execution_classes = find_adapter_classes()
        # for keeping track of our subprocess id for later termination
        self.subprocess_id = subprocess_id
        self.settings = settings
        self.envs = envs
        self.include_world_processing = include_world_processing

        if self.envs and len(self.envs.keys()) > 0:
            self.initialized_classes = [init_adapter_class(cls[0], self.envs) for cls in self.execution_classes if init_adapter_class(cls[0], self.envs) is not None]
            
                

    def _save(self):
        #get existing agent in database.
        session = Session
        session.query(agent).filter_by(id = self.agent_id).first()
        agent.world_context = self.world_context
        agent.task_context = self.task_context
        agent.execution_context = self.task_context
        session.commit()


    