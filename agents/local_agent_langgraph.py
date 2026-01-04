"""
LangGraph-enabled LocalAgent implementation.

This module extends the LocalAgent with LangGraph StateGraph functionality
for automatic checkpointing and persistence.
"""
import logging
from typing import Optional, Dict, Any
from agents.local_agent import LocalAgent
from agents.langgraph_agent_mixin import LangGraphAgentMixin
from agents.agent_context import AgentContext

logger = logging.getLogger(__name__)


class LocalAgentLangGraph(LangGraphAgentMixin, LocalAgent):
    """
    LocalAgent with LangGraph StateGraph execution and persistence.

    This class combines the local inference functionality of LocalAgent
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
        model_id: str,
        runner_type: str = "mlx_llama",
        device_config: Optional[Dict[str, Any]] = None,
        as_subprocess: bool = False,
        **kwargs
    ):
        """
        Initialize LocalAgentLangGraph.

        Args:
            agent_context: Agent context object
            model_id: Model identifier for local inference
            runner_type: Type of runner to use (e.g., "mlx_llama", "vllm")
            device_config: Device configuration for the runner
            as_subprocess: Whether to run as subprocess
        """
        # Initialize both parent classes
        # The MRO ensures LangGraphAgentMixin.__init__ is called first
        super().__init__(
            agent_context=agent_context,
            model_id=model_id,
            runner_type=runner_type,
            device_config=device_config,
            as_subprocess=as_subprocess,
            **kwargs
        )

        self.logger.info(f"LocalAgentLangGraph initialized with model {model_id}")

    def phase_out(self):
        """
        Phase out the agent with LangGraph checkpoint persistence.

        This method:
        1. Saves the current state to a checkpoint
        2. Persists to the database
        3. Cleans up runner resources
        """
        try:
            self.logger.info(f"Phasing out agent {self.thread_id}")

            # Get final state and ensure it's checkpointed
            final_state = self._sync_context_to_state()

            # The checkpointer will automatically save during the last tick,
            # but we ensure context is saved to database
            self._agent_context._save()

            # Cleanup runner resources
            if hasattr(self, 'runner') and self.runner and hasattr(self.runner, 'cleanup'):
                self.runner.cleanup()

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

        This overrides both LocalAgent.tick() and uses the LangGraph
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
