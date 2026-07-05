import logging
import uuid
from typing import Any, Dict, List, Optional

from adapters.adapter_registry import find_adapter_classes, init_adapter_class
from agents.agent_settings import AgentSettings
from app.models.database.agent import Agent
from app.models.database.chat_session import ChatSession, update_chat_history
from app.models.database.database import SessionLocal

import json

logger = logging.getLogger(__name__)


class AgentContext():
    def __init__(self, settings: AgentSettings, agent_id: Optional[str] = None,
                 world_context: Optional[List[str]] = None,
                 task_context: Optional[List[str]] = None,
                 execution_context: Optional[List[str]] = None,
                 function_log: Optional[List[str]] = None,
                 execution_classes: Optional[List[Any]] = None,
                 subprocess_id: Optional[int] = None,
                 envs: Optional[Dict[str, str]] = None,
                 include_world_processing: bool = False):
        self.agent_id = agent_id if agent_id is not None else str(uuid.uuid4())
        self.world_context = world_context if world_context is not None else []
        self.task_context = task_context if task_context is not None else []
        self.execution_context = execution_context if execution_context is not None else []
        self.function_log = function_log if function_log is not None else []
        self.execution_classes = execution_classes if execution_classes is not None else []
        if not self.execution_classes:
            #if self.execution_classes are not defined, pull from adapter_registry
            self.execution_classes = find_adapter_classes()
        # for keeping track of our subprocess id for later termination
        self.subprocess_id = subprocess_id
        self.settings = settings
        self.envs = envs if envs is not None else {}
        self.include_world_processing = include_world_processing

        self.initialized_classes = []
        if self.envs:
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

    def _add_to_chat_history(self, user_message: str, ai_message: str, chat_id: Optional[int] = None) -> ChatSession:
        return update_chat_history(session_id=chat_id, new_user_message=user_message, new_ai_message=ai_message)
