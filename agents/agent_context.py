from app.models.database import database
from agents.agent_settings import AgentSettings
from typing import List, Any, Dict
from app.models.database.agent import Agent
from app.models.database.database import Base, SessionLocal
import uuid
import json
import logging
from adapters.adapter_registry import find_adapter_classes, init_adapter_class
from adapters.tool_schema import ToolSchema, enumerate_tool_schemas
from agents.tool_calling import ToolDispatcher
from app.models.database.chat_session import get_chat_history, update_chat_history, ChatSession
from app.models.database.agent_snapshot import (
    create_snapshot,
    get_latest_snapshot,
    get_snapshot_by_id,
    prune_snapshots_older_than,
)

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

        self.initialized_classes = []
        if self.envs and len(self.envs.keys()) > 0:
            # Fix: avoid calling init_adapter_class twice by storing result first
            for cls in self.execution_classes:
                initialized = init_adapter_class(cls[0], self.envs)
                if initialized is not None:
                    self.initialized_classes.append(initialized)

        # Lazily-built caches for structured tool calling
        self._tool_schemas = None
        self._tool_dispatcher = None

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

    def get_tool_schemas(self) -> List[ToolSchema]:
        """
        Reflected JSON schemas for every action of the initialized adapters.

        These give models full function visibility: adapter/action names, typed
        parameters, required arguments, and docstring descriptions.
        """
        if self._tool_schemas is None:
            schemas = []
            for wrapper in self.initialized_classes:
                try:
                    schemas.extend(enumerate_tool_schemas(wrapper.instance))
                except Exception as e:
                    logger.warning(f"Failed to build tool schemas for adapter {wrapper.name}: {e}")
            self._tool_schemas = schemas
        return self._tool_schemas

    def get_tool_dispatcher(self) -> ToolDispatcher:
        """Dispatcher that validates and executes tool calls, journaling to function_log."""
        if self._tool_dispatcher is None:
            self._tool_dispatcher = ToolDispatcher(
                self.initialized_classes,
                schemas=self.get_tool_schemas(),
                function_log=self.function_log,
            )
        return self._tool_dispatcher

    def snapshot(self, reason: str = "manual"):
        """
        Persist an append-only snapshot of the current execution state.

        Failures are logged rather than raised so snapshotting can never break
        an agent loop. Returns the created AgentSnapshot or None on failure.
        """
        try:
            snapshot = create_snapshot(
                agent_identifier=self.agent_id,
                world_context=self.world_context,
                task_context=self.task_context,
                execution_context=self.execution_context,
                function_log=self.function_log,
                reason=reason,
            )
            logger.info(
                f"Saved snapshot step={snapshot.step} reason={reason} for agent_id={self.agent_id}"
            )
        except Exception as e:
            logger.error(f"Failed to snapshot agent context for agent_id={self.agent_id}: {e}")
            return None

        # Retention: expire old snapshots on write so history is bounded without
        # a separate maintenance job. Prune failures are as harmless as snapshot
        # failures — log and move on.
        try:
            retention_days = getattr(self.settings, "snapshot_retention_days", 7)
            pruned = prune_snapshots_older_than(self.agent_id, retention_days)
            if pruned:
                logger.info(
                    f"Pruned {pruned} snapshot(s) older than {retention_days} days for agent_id={self.agent_id}"
                )
        except Exception as e:
            logger.warning(f"Failed to prune old snapshots for agent_id={self.agent_id}: {e}")

        return snapshot

    def restore_snapshot(self, snapshot_id: int = None) -> bool:
        """
        Rehydrate execution state from a snapshot.

        Restores the latest snapshot for this agent, or a specific snapshot_id
        (which must belong to this agent). Returns True if state was restored.
        """
        try:
            if snapshot_id is not None:
                snapshot = get_snapshot_by_id(snapshot_id)
                if snapshot is not None and snapshot.agent_identifier != self.agent_id:
                    logger.warning(
                        f"Snapshot {snapshot_id} belongs to agent {snapshot.agent_identifier}, "
                        f"not {self.agent_id}; refusing to restore"
                    )
                    return False
            else:
                snapshot = get_latest_snapshot(self.agent_id)
        except Exception as e:
            logger.error(f"Failed to load snapshot for agent_id={self.agent_id}: {e}")
            return False

        if snapshot is None:
            logger.info(f"No snapshot found for agent_id={self.agent_id}")
            return False

        state = snapshot.to_dict()
        self.world_context = state["world_context"]
        self.task_context = state["task_context"]
        self.execution_context = state["execution_context"]
        self.function_log = state["function_log"]
        # function_log identity changed; rebuild the dispatcher so journaling follows it
        self._tool_dispatcher = None
        logger.info(
            f"Restored snapshot step={state['step']} for agent_id={self.agent_id}"
        )
        return True


    