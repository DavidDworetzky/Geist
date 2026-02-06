from app.models.database import database
from agents.agent_settings import AgentSettings
from typing import List, Any, Dict
from app.models.database.agent import Agent
from app.models.database.database import Base, SessionLocal
import uuid
import json
import logging
from adapters.adapter_registry import find_adapter_classes, init_adapter_class
from app.models.database.chat_session import get_chat_history, update_chat_history, ChatSession

logger = logging.getLogger(__name__)
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
            # Fix: avoid calling init_adapter_class twice by storing result first
            self.initialized_classes = []
            for cls in self.execution_classes:
                initialized = init_adapter_class(cls[0], self.envs)
                if initialized is not None:
                    self.initialized_classes.append(initialized)
            
                

    def _save(self):
        """Save agent context to database."""
        try:
            with SessionLocal() as session:
                # Query for existing agent
                agent_obj = session.query(Agent).filter_by(agent_id=self.agent_id).first()
                if agent_obj:
                    # Update existing agent context (serialize lists to JSON)
                    agent_obj.world_context = json.dumps(self.world_context)
                    agent_obj.task_context = json.dumps(self.task_context)
                    agent_obj.execution_context = json.dumps(self.execution_context)
                    session.commit()
                    logger.info(f"Saved agent context for agent_id={self.agent_id}")
                else:
                    logger.warning(f"Agent with agent_id={self.agent_id} not found in database, skipping save")
        except Exception as e:
            logger.error(f"Failed to save agent context: {e}")

    def _add_to_chat_history(self, user_message: str, ai_message: str, chat_id: int = None) -> ChatSession:
        chat_history = get_chat_history(chat_id)
        chat_history.append({"user": user_message, "ai": ai_message})
        return update_chat_history(session_id = chat_id, new_user_message = user_message, new_ai_message = ai_message)


    