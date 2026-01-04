"""
LangGraph state schema definitions for agent persistence.

This module defines the TypedDict schemas used by LangGraph for state management
and checkpointing in the agent execution loop.
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class AgentState(TypedDict):
    """
    State schema for the agent execution loop.

    This TypedDict defines the structure of the agent's state that will be
    managed by LangGraph's StateGraph and persisted via checkpoints.

    Attributes:
        agent_id: Unique identifier for the agent
        world_context: List of facts about the world state
        task_context: List of task decomposition items
        execution_context: List of current execution tasks
        function_log: History of function calls made
        current_tick: Current tick number for tracking execution progress
        last_completion: Last completion result from the model
        error: Any error that occurred during execution
        metadata: Additional metadata for tracking (e.g., timestamps, model info)
    """
    agent_id: str
    world_context: Annotated[List[str], operator.add]
    task_context: Annotated[List[str], operator.add]
    execution_context: Annotated[List[str], operator.add]
    function_log: Annotated[List[str], operator.add]
    current_tick: int
    last_completion: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]


class CheckpointMetadata(TypedDict):
    """
    Metadata for checkpoints.

    Attributes:
        source: Source of the checkpoint (e.g., "tick_world", "tick_tasks", "tick_execution")
        timestamp: ISO format timestamp of checkpoint creation
        step: Step number in the execution sequence
        parent_config: Parent checkpoint configuration for traversal
    """
    source: str
    timestamp: str
    step: int
    parent_config: Optional[Dict[str, Any]]
