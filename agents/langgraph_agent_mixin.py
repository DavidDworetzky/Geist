"""
LangGraph agent mixin for state graph execution and persistence.

This module provides a mixin class that adds LangGraph StateGraph functionality
to existing agents, enabling automatic checkpointing and state persistence.
"""
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from agents.langgraph_state import AgentState
from agents.langgraph_checkpointer import SQLAlchemyCheckpointSaver

logger = logging.getLogger(__name__)


class LangGraphAgentMixin:
    """
    Mixin class that adds LangGraph StateGraph execution to agents.

    This mixin converts the traditional tick-based execution loop into a
    LangGraph StateGraph with automatic checkpointing and persistence.

    Usage:
        class MyAgent(LangGraphAgentMixin, BaseAgent):
            ...
    """

    def __init__(self, *args, **kwargs):
        """Initialize the LangGraph mixin."""
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.checkpointer: Optional[BaseCheckpointSaver] = None
        self.graph = None
        self.thread_id = None
        self._setup_langgraph()

    def _setup_langgraph(self):
        """Set up the LangGraph StateGraph and checkpointer."""
        try:
            # Initialize checkpointer
            self.checkpointer = SQLAlchemyCheckpointSaver()

            # Create the state graph
            workflow = StateGraph(AgentState)

            # Add nodes for each tick type
            workflow.add_node("tick_world", self._langgraph_tick_world)
            workflow.add_node("tick_tasks", self._langgraph_tick_tasks)
            workflow.add_node("tick_execution", self._langgraph_tick_execution)

            # Define the flow
            # Start -> World (if enabled) -> Tasks -> Execution -> End
            if self._agent_context.settings.include_world_processing:
                workflow.set_entry_point("tick_world")
                workflow.add_edge("tick_world", "tick_tasks")
            else:
                workflow.set_entry_point("tick_tasks")

            workflow.add_edge("tick_tasks", "tick_execution")
            workflow.add_edge("tick_execution", END)

            # Compile the graph with checkpointer
            self.graph = workflow.compile(checkpointer=self.checkpointer)

            # Set thread ID for this agent instance
            self.thread_id = self._agent_context.agent_id

            self.logger.info(f"LangGraph StateGraph initialized for agent {self.thread_id}")

        except Exception as e:
            self.logger.error(f"Failed to setup LangGraph: {e}")
            raise

    def _langgraph_tick_world(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node for world context tick.

        Args:
            state: Current agent state

        Returns:
            Updated state dict
        """
        try:
            self.logger.debug(f"[LangGraph] tick_world for agent {state['agent_id']}")

            # Update agent context from state
            self._sync_state_to_context(state)

            # Execute the original tick_world method
            result = self.tick_world()

            # Update state with new world context
            return {
                "world_context": result if result else state["world_context"],
                "current_tick": state["current_tick"] + 1,
                "metadata": {
                    **state.get("metadata", {}),
                    "last_world_tick": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Error in tick_world node: {e}")
            return {
                "error": str(e),
                "metadata": {
                    **state.get("metadata", {}),
                    "last_error": datetime.now().isoformat()
                }
            }

    def _langgraph_tick_tasks(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node for task context tick.

        Args:
            state: Current agent state

        Returns:
            Updated state dict
        """
        try:
            self.logger.debug(f"[LangGraph] tick_tasks for agent {state['agent_id']}")

            # Update agent context from state
            self._sync_state_to_context(state)

            # Execute the original tick_tasks method
            result = self.tick_tasks()

            # Update state with new task context
            return {
                "task_context": result if result else state["task_context"],
                "current_tick": state["current_tick"] + 1,
                "metadata": {
                    **state.get("metadata", {}),
                    "last_tasks_tick": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Error in tick_tasks node: {e}")
            return {
                "error": str(e),
                "metadata": {
                    **state.get("metadata", {}),
                    "last_error": datetime.now().isoformat()
                }
            }

    def _langgraph_tick_execution(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node for execution context tick.

        Args:
            state: Current agent state

        Returns:
            Updated state dict
        """
        try:
            self.logger.debug(f"[LangGraph] tick_execution for agent {state['agent_id']}")

            # Update agent context from state
            self._sync_state_to_context(state)

            # Execute the original tick_execution method
            result = self.tick_execution()

            # Update state with new execution context
            return {
                "execution_context": result if result else state["execution_context"],
                "current_tick": state["current_tick"] + 1,
                "metadata": {
                    **state.get("metadata", {}),
                    "last_execution_tick": datetime.now().isoformat()
                }
            }

        except Exception as e:
            self.logger.error(f"Error in tick_execution node: {e}")
            return {
                "error": str(e),
                "metadata": {
                    **state.get("metadata", {}),
                    "last_error": datetime.now().isoformat()
                }
            }

    def _sync_state_to_context(self, state: AgentState):
        """
        Synchronize LangGraph state to AgentContext.

        Args:
            state: Current agent state from LangGraph
        """
        self._agent_context.world_context = state.get("world_context", [])
        self._agent_context.task_context = state.get("task_context", [])
        self._agent_context.execution_context = state.get("execution_context", [])
        self._agent_context.function_log = state.get("function_log", [])

    def _sync_context_to_state(self) -> AgentState:
        """
        Synchronize AgentContext to LangGraph state.

        Returns:
            AgentState dict representing current context
        """
        return {
            "agent_id": self._agent_context.agent_id,
            "world_context": self._agent_context.world_context or [],
            "task_context": self._agent_context.task_context or [],
            "execution_context": self._agent_context.execution_context or [],
            "function_log": self._agent_context.function_log or [],
            "current_tick": 0,
            "last_completion": None,
            "error": None,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "agent_type": self.__class__.__name__,
            }
        }

    def langgraph_tick(self) -> Dict[str, Any]:
        """
        Execute one tick cycle through the LangGraph StateGraph.

        This method replaces the traditional tick() method and provides
        automatic checkpointing at each step.

        Returns:
            Final state after tick execution
        """
        try:
            # Get current state from context
            initial_state = self._sync_context_to_state()

            # Configure the graph invocation with thread_id
            config = {
                "configurable": {
                    "thread_id": self.thread_id,
                    "checkpoint_ns": "",
                }
            }

            # Invoke the graph
            self.logger.info(f"Invoking LangGraph tick for agent {self.thread_id}")
            final_state = self.graph.invoke(initial_state, config=config)

            # Sync final state back to context
            self._sync_state_to_context(final_state)

            # Save to database (legacy save method)
            self._agent_context._save()

            self.logger.info(f"LangGraph tick completed for agent {self.thread_id}")

            return final_state

        except Exception as e:
            self.logger.error(f"Error during LangGraph tick: {e}")
            raise

    def langgraph_stream_tick(self):
        """
        Stream the tick execution step by step.

        Yields intermediate states as each node executes.
        """
        try:
            initial_state = self._sync_context_to_state()

            config = {
                "configurable": {
                    "thread_id": self.thread_id,
                    "checkpoint_ns": "",
                }
            }

            self.logger.info(f"Streaming LangGraph tick for agent {self.thread_id}")

            for step in self.graph.stream(initial_state, config=config):
                self.logger.debug(f"LangGraph step: {step}")
                yield step

            # Sync final state back after streaming
            final_state = self.graph.get_state(config)
            if final_state and hasattr(final_state, 'values'):
                self._sync_state_to_context(final_state.values)

            self._agent_context._save()

        except Exception as e:
            self.logger.error(f"Error during LangGraph stream tick: {e}")
            raise

    def langgraph_get_state(self) -> Optional[AgentState]:
        """
        Get the current checkpointed state from LangGraph.

        Returns:
            Current agent state or None if not found
        """
        try:
            config = {
                "configurable": {
                    "thread_id": self.thread_id,
                    "checkpoint_ns": "",
                }
            }

            state_snapshot = self.graph.get_state(config)
            if state_snapshot and hasattr(state_snapshot, 'values'):
                return state_snapshot.values

            return None

        except Exception as e:
            self.logger.error(f"Error getting LangGraph state: {e}")
            return None

    def langgraph_restore_from_checkpoint(self):
        """
        Restore agent state from the latest checkpoint.

        This is used during phase_in to rehydrate the agent.
        """
        try:
            self.logger.info(f"Restoring agent {self.thread_id} from checkpoint")

            state = self.langgraph_get_state()

            if state:
                self._sync_state_to_context(state)
                self.logger.info(f"Agent {self.thread_id} restored from checkpoint")
            else:
                self.logger.warning(f"No checkpoint found for agent {self.thread_id}")

        except Exception as e:
            self.logger.error(f"Error restoring from checkpoint: {e}")
            raise

    def tick(self):
        """
        Override the base tick() method to use LangGraph execution.

        This maintains backward compatibility while using the new
        LangGraph-based execution model.
        """
        self.langgraph_tick()
