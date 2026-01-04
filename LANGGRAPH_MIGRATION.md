# LangGraph Agent Persistence Migration

This document describes the migration of Geist agents to use LangGraph/Langchain for agent persistence and state management.

## Overview

The core execution loop of `OnlineAgent`, `LocalAgent`, and `BaseAgent` has been migrated to use **LangGraph StateGraph** for automatic checkpointing and persistence. This provides:

- ✅ **Automatic State Persistence**: Every tick is automatically checkpointed to the database
- ✅ **Resume from Checkpoint**: Agents can be phased out and phased in, restoring from the last checkpoint
- ✅ **Thread-based Execution**: Each agent instance has a unique thread ID for tracking
- ✅ **Human-in-the-Loop**: Foundation for future interactive agent flows
- ✅ **Time Travel Debugging**: Access historical states for debugging
- ✅ **Fault Tolerance**: Recover from failures by resuming from last checkpoint

## Architecture Changes

### Before: Tick-based Execution

```python
class BaseAgent(ABC):
    def tick(self):
        if self._agent_context.settings.include_world_processing:
            self.tick_world()
        self.tick_tasks()
        self.tick_execution()
```

### After: LangGraph StateGraph

```python
class LangGraphAgentMixin:
    def _setup_langgraph(self):
        workflow = StateGraph(AgentState)

        # Add nodes for each tick type
        workflow.add_node("tick_world", self._langgraph_tick_world)
        workflow.add_node("tick_tasks", self._langgraph_tick_tasks)
        workflow.add_node("tick_execution", self._langgraph_tick_execution)

        # Define the flow with automatic checkpointing
        workflow.set_entry_point("tick_tasks")
        workflow.add_edge("tick_tasks", "tick_execution")
        workflow.add_edge("tick_execution", END)

        # Compile with checkpointer
        self.graph = workflow.compile(checkpointer=self.checkpointer)
```

## New Components

### 1. `langgraph_state.py` - State Schema

Defines the TypedDict schemas for agent state:

```python
class AgentState(TypedDict):
    agent_id: str
    world_context: Annotated[List[str], operator.add]
    task_context: Annotated[List[str], operator.add]
    execution_context: Annotated[List[str], operator.add]
    function_log: Annotated[List[str], operator.add]
    current_tick: int
    last_completion: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]
```

### 2. `langgraph_checkpointer.py` - Database Persistence

Custom `BaseCheckpointSaver` implementation that integrates with the existing SQLAlchemy database:

```python
class SQLAlchemyCheckpointSaver(BaseCheckpointSaver):
    def put(self, config, checkpoint, metadata):
        # Save checkpoint to Agent table

    def get_tuple(self, config):
        # Retrieve checkpoint from Agent table

    def list(self, config, filter, before, limit):
        # List available checkpoints
```

### 3. `langgraph_agent_mixin.py` - StateGraph Execution

Mixin class that adds LangGraph functionality to existing agents:

- Converts tick methods to graph nodes
- Manages state synchronization
- Provides checkpoint restore functionality
- Maintains backward compatibility

### 4. New Agent Classes

- **`OnlineAgentLangGraph`**: LangGraph-enabled OnlineAgent
- **`LocalAgentLangGraph`**: LangGraph-enabled LocalAgent

### 5. Database Migration

New fields added to the `Agent` table:

```sql
ALTER TABLE agent ADD COLUMN checkpoint_data TEXT;
ALTER TABLE agent ADD COLUMN checkpoint_metadata TEXT;
ALTER TABLE agent ADD COLUMN pending_writes TEXT;
```

Migration file: `migrations/versions/a1b2c3d4e5f6_add_langgraph_checkpoint_fields.py`

## Usage

### Creating LangGraph-Enabled Agents

The `AgentFactory` now supports a `use_langgraph` parameter (defaults to `True`):

```python
from agents.factory import AgentFactory
from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings

# Create context
settings = AgentSettings(name="MyAgent", version="1.0", description="Test agent")
context = AgentContext(settings=settings)

# Create LangGraph-enabled agent (default)
agent = AgentFactory.create_agent(
    agent_type="online",
    agent_context=context,
    model="gpt-4",
    use_langgraph=True  # This is the default
)

# Or create traditional agent (legacy)
legacy_agent = AgentFactory.create_agent(
    agent_type="online",
    agent_context=context,
    model="gpt-4",
    use_langgraph=False
)
```

### Executing Ticks

```python
# Standard tick execution (with automatic checkpointing)
agent.tick()

# Stream tick execution (see intermediate states)
for step in agent.stream_tick():
    print(f"Step: {step}")

# Get current checkpoint state
state = agent.get_checkpoint_state()
print(f"Current state: {state}")
```

### Phase Out and Phase In

```python
# Phase out agent (saves checkpoint and cleans up resources)
agent.phase_out()

# Later... phase in agent (restores from checkpoint)
agent.phase_in()

# Continue execution from last checkpoint
agent.tick()
```

### Direct Access to LangGraph Methods

```python
# Execute one tick through the graph
final_state = agent.langgraph_tick()

# Stream execution step by step
for step in agent.langgraph_stream_tick():
    print(step)

# Get current checkpointed state
state = agent.langgraph_get_state()

# Restore from checkpoint
agent.langgraph_restore_from_checkpoint()
```

## Migration Guide

### For Existing Code

**Option 1: Use LangGraph by Default (Recommended)**

No code changes needed! The factory defaults to `use_langgraph=True`.

```python
# This now creates a LangGraph-enabled agent
agent = AgentFactory.create_agent(
    agent_type="online",
    agent_context=context,
    model="gpt-4"
)
```

**Option 2: Explicitly Disable LangGraph**

If you need to use the traditional agents:

```python
agent = AgentFactory.create_agent(
    agent_type="online",
    agent_context=context,
    model="gpt-4",
    use_langgraph=False
)
```

### For Custom Agent Implementations

If you have custom agents that extend `OnlineAgent` or `LocalAgent`:

1. **Option A**: Extend the LangGraph versions

```python
from agents.online_agent_langgraph import OnlineAgentLangGraph

class MyCustomAgent(OnlineAgentLangGraph):
    # Your custom implementation
    pass
```

2. **Option B**: Add the mixin to your agent

```python
from agents.langgraph_agent_mixin import LangGraphAgentMixin
from agents.online_agent import OnlineAgent

class MyCustomAgent(LangGraphAgentMixin, OnlineAgent):
    # Your custom implementation
    pass
```

### Database Migration

Run the Alembic migration to add the checkpoint fields:

```bash
# Apply the migration
alembic upgrade head

# Or if using Docker
docker compose exec geist alembic upgrade head
```

### Dependencies

Install the new dependencies (already added to environment files):

```bash
pip install langchain==0.3.13 langchain-core==0.3.28 langgraph==0.2.62 langgraph-checkpoint==2.0.6

# Or update conda environment
conda env update -f linux_environment.yml
```

## Benefits

### 1. Automatic Persistence

Every node execution is automatically checkpointed. No need to manually call `_save()`.

### 2. Resume from Failures

If an agent crashes or is terminated, it can resume from the last checkpoint:

```python
agent = AgentFactory.create_agent(agent_type="online", agent_context=context)
agent.phase_in()  # Restores from last checkpoint
agent.tick()      # Continues execution
```

### 3. State Inspection

Access the full state at any point:

```python
state = agent.get_checkpoint_state()
print(f"World Context: {state['world_context']}")
print(f"Task Context: {state['task_context']}")
print(f"Execution Context: {state['execution_context']}")
print(f"Current Tick: {state['current_tick']}")
```

### 4. Time Travel Debugging

List all checkpoints for an agent:

```python
checkpointer = SQLAlchemyCheckpointSaver()
config = {"configurable": {"thread_id": agent.thread_id}}
for checkpoint in checkpointer.list(config, limit=10):
    print(f"Checkpoint at {checkpoint.metadata['timestamp']}")
```

### 5. Foundation for Advanced Features

- **Human-in-the-Loop**: Pause execution and wait for human approval
- **Branching**: Create alternative execution paths
- **Replay**: Re-execute from any checkpoint
- **Multi-Agent**: Coordinate multiple agents with shared state

## Backward Compatibility

The migration maintains full backward compatibility:

- ✅ All existing `tick()`, `tick_world()`, `tick_tasks()`, `tick_execution()` methods work unchanged
- ✅ `AgentContext` API remains the same
- ✅ `phase_out()` and `phase_in()` behavior is preserved
- ✅ Traditional agents can still be created with `use_langgraph=False`

## Implementation Details

### State Synchronization

The mixin automatically synchronizes between `AgentContext` and `AgentState`:

```python
def _sync_state_to_context(self, state: AgentState):
    """LangGraph state -> AgentContext"""
    self._agent_context.world_context = state["world_context"]
    self._agent_context.task_context = state["task_context"]
    self._agent_context.execution_context = state["execution_context"]

def _sync_context_to_state(self) -> AgentState:
    """AgentContext -> LangGraph state"""
    return {
        "agent_id": self._agent_context.agent_id,
        "world_context": self._agent_context.world_context,
        # ... etc
    }
```

### Checkpoint Storage

Checkpoints are stored as JSON in the `Agent` table:

```python
{
    "checkpoint_data": {
        "v": 1,
        "id": "checkpoint_id",
        "channel_values": {
            "world_context": [...],
            "task_context": [...],
            "execution_context": [...]
        },
        "channel_versions": {...},
        "versions_seen": {...}
    },
    "checkpoint_metadata": {
        "source": "tick_execution",
        "step": 5,
        "writes": {...}
    }
}
```

### Thread Management

Each agent instance uses its `agent_id` as the `thread_id` for LangGraph:

```python
self.thread_id = self._agent_context.agent_id

config = {
    "configurable": {
        "thread_id": self.thread_id,
        "checkpoint_ns": "",
    }
}

final_state = self.graph.invoke(initial_state, config=config)
```

## Testing

### Manual Testing

```python
# Create agent
agent = AgentFactory.create_agent(
    agent_type="online",
    agent_context=context,
    model="gpt-4"
)

# Initialize with task
agent.initialize(task_prompt="Write a Python function to calculate factorial")

# Execute a few ticks
agent.tick()
agent.tick()

# Check state
state = agent.get_checkpoint_state()
assert state['current_tick'] >= 2

# Phase out
agent.phase_out()

# Recreate and phase in
new_agent = AgentFactory.create_agent(
    agent_type="online",
    agent_context=context,  # Same context (agent_id)
    model="gpt-4"
)
new_agent.phase_in()

# Verify state was restored
restored_state = new_agent.get_checkpoint_state()
assert restored_state['task_context'] == state['task_context']
```

### Unit Tests

TODO: Add unit tests for:
- [ ] Checkpoint saving and restoration
- [ ] State synchronization
- [ ] Phase out/phase in cycle
- [ ] Error handling in graph nodes
- [ ] Streaming execution

## Future Enhancements

### Planned Features

1. **Human-in-the-Loop Nodes**
   - Add interrupt nodes that wait for human approval
   - Enable interactive debugging and steering

2. **Conditional Branching**
   - Use LangGraph's conditional edges for dynamic flows
   - Enable different execution paths based on state

3. **Multi-Agent Coordination**
   - Share state across multiple agents
   - Coordinate complex multi-agent workflows

4. **Advanced Checkpointing**
   - Snapshot-based rollback
   - Checkpoint compression
   - S3/cloud storage backends

5. **Monitoring and Observability**
   - LangSmith integration
   - Execution trace visualization
   - Performance metrics

## Troubleshooting

### Issue: `ImportError: cannot import name 'StateGraph'`

**Solution**: Install LangGraph dependencies

```bash
pip install langgraph==0.2.62 langgraph-checkpoint==2.0.6
```

### Issue: `DatabaseError: column checkpoint_data does not exist`

**Solution**: Run the Alembic migration

```bash
alembic upgrade head
```

### Issue: Agent doesn't restore state on `phase_in()`

**Possible causes**:
1. Check that the agent_id is the same between phase_out and phase_in
2. Verify the checkpoint was saved to the database
3. Check database logs for errors

**Debug**:

```python
# Check if checkpoint exists
from agents.langgraph_checkpointer import SQLAlchemyCheckpointSaver
checkpointer = SQLAlchemyCheckpointSaver()
config = {"configurable": {"thread_id": agent.thread_id}}
checkpoint = checkpointer.get_tuple(config)
print(f"Checkpoint found: {checkpoint is not None}")
```

### Issue: Performance degradation

**Solution**: The checkpointer writes to the database on every node execution. For high-frequency agents, consider:

1. Batching checkpoint writes
2. Using an in-memory checkpointer for development
3. Optimizing database indices

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Persistence Guide](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Checkpoint API](https://reference.langchain.com/python/langgraph/checkpoints/)

## Questions?

For questions or issues related to this migration:
- Check the [GitHub Issues](https://github.com/YourOrg/Geist/issues)
- Review the source code in `/agents/langgraph_*` files
- Consult the LangGraph documentation

---

**Migration completed**: 2026-01-04
**LangGraph version**: 0.2.62
**Langchain version**: 0.3.13
