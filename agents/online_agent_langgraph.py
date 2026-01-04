"""
LangGraph-enabled OnlineAgent implementation.

This module extends the OnlineAgent with LangGraph StateGraph functionality
for automatic checkpointing and persistence.
"""
import logging
from typing import Optional, Dict, Any, List
from agents.online_agent import OnlineAgent
from agents.langgraph_agent_mixin import LangGraphAgentMixin
from agents.agent_context import AgentContext

logger = logging.getLogger(__name__)


class OnlineAgentLangGraph(LangGraphAgentMixin, OnlineAgent):
    """
    OnlineAgent with LangGraph StateGraph execution and persistence.

    This class combines the HTTP-based API calling functionality of OnlineAgent
    with LangGraph's StateGraph for automatic checkpointing and state management.

    The execution loop is now managed by a StateGraph with the following nodes:
    - tick_world: Updates world context reasoning
    - tick_tasks: Decomposes tasks into actionable subtasks
    - tick_execution: Generates function calls for task execution

    Each node execution is automatically checkpointed to the database.
    """

    def __init__(
        self,
        agent_context: AgentContext,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        backup_providers: Optional[List[Dict[str, str]]] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        as_subprocess: bool = False,
        **kwargs
    ):
        """
        Initialize OnlineAgentLangGraph.

        Args:
            agent_context: Agent context object
            base_url: Base URL for the API endpoint
            model: Model name to use
            api_key: API key for authentication
            backup_providers: List of backup provider configurations
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            as_subprocess: Whether to run as subprocess
        """
        # Initialize both parent classes
        # The MRO (Method Resolution Order) ensures LangGraphAgentMixin.__init__
        # is called first, which then calls OnlineAgent.__init__ via super()
        super().__init__(
            agent_context=agent_context,
            base_url=base_url,
            model=model,
            api_key=api_key,
            backup_providers=backup_providers,
            timeout=timeout,
            max_retries=max_retries,
            as_subprocess=as_subprocess,
            **kwargs
        )

        self.logger.info(f"OnlineAgentLangGraph initialized with model {model}")

    def phase_out(self):
        """
        Phase out the agent with LangGraph checkpoint persistence.

        This method:
        1. Saves the current state to a checkpoint
        2. Persists to the database
        3. Cleans up resources
        """
        try:
            self.logger.info(f"Phasing out agent {self.thread_id}")

            # Get final state and ensure it's checkpointed
            final_state = self._sync_context_to_state()

            # The checkpointer will automatically save during the last tick,
            # but we ensure context is saved to database
            self._agent_context._save()

            # Close HTTP client
            if hasattr(self, 'client'):
                self.client.close()

            # Terminate subprocess if running
            self.terminate_subprocess()

            self.logger.info(f"Agent {self.thread_id} phased out successfully")

        except Exception as e:
            self.logger.error(f"Error during phase_out: {e}")
            raise

    def phase_in(self):
        """
        Phase in the agent by restoring from checkpoint.

        This method:
        1. Restores the agent state from the latest checkpoint
        2. Rehydrates the agent context
        3. Re-initializes resources
        """
        try:
            self.logger.info(f"Phasing in agent {self.thread_id}")

            # Restore from checkpoint
            self.langgraph_restore_from_checkpoint()

            # Re-initialize the agent (but don't add new task)
            self.initialize(task_prompt=None)

            self.logger.info(f"Agent {self.thread_id} phased in successfully")

        except Exception as e:
            self.logger.error(f"Error during phase_in: {e}")
            raise

    def tick(self):
        """
        Execute one agent tick using LangGraph StateGraph.

        This overrides both OnlineAgent.tick() and uses the LangGraph
        execution model with automatic checkpointing.
        """
        return self.langgraph_tick()

    def stream_tick(self):
        """
        Stream the tick execution step by step.

        Yields intermediate states as each node executes, useful for
        monitoring and debugging.
        """
        return self.langgraph_stream_tick()

    def get_checkpoint_state(self):
        """
        Get the current checkpointed state.

        Returns:
            Current agent state from the latest checkpoint
        """
        return self.langgraph_get_state()
