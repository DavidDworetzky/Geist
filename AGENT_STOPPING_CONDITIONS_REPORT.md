# Agent Stopping Conditions Architecture Report

## Executive Summary

This report analyzes the current Geist `BaseAgent` architecture and proposes a design for implementing **stopping conditions** for agent task execution. We compare this to standard agent frameworks like the **Anthropic Claude SDK** and **OpenAI Agent patterns** to identify best practices and implementation strategies.

**Current State**: Geist agents use a manual, single-tick execution model with no built-in stopping conditions.

**Proposed State**: Add a robust agent execution loop with configurable stopping conditions similar to modern agent frameworks.

---

## Table of Contents

1. [Current BaseAgent Architecture](#current-baseagent-architecture)
2. [Stopping Condition Patterns in Modern SDKs](#stopping-condition-patterns)
3. [Proposed Architecture](#proposed-architecture)
4. [Implementation Comparison](#implementation-comparison)
5. [Migration Path](#migration-path)
6. [Code Examples](#code-examples)

---

## Current BaseAgent Architecture

### Overview

The Geist agent system (`/home/user/Geist/agents/base_agent.py`) follows a **tick-based execution model** with three reasoning phases:

```python
class BaseAgent(ABC):
    def tick(self):
        if self._agent_context.settings.include_world_processing:
            self.tick_world()      # Phase 1: World state reasoning
        self.tick_tasks()          # Phase 2: Task decomposition
        self.tick_execution()      # Phase 3: Function execution
```

### Key Characteristics

| Aspect | Current Implementation |
|--------|----------------------|
| **Execution Model** | Single-tick, manually triggered via API |
| **API Endpoint** | `POST /agent/initialize_task_and_tick` |
| **Iteration Control** | None (one tick per API call) |
| **Stopping Conditions** | None |
| **State Management** | Context accumulation in lists |
| **Function Calling** | JSON-based, dynamic adapter invocation |
| **Error Handling** | Retry logic (max 3 attempts) for invalid JSON |

### Execution Flow

```
User Request → API Endpoint
    ↓
Initialize Task → agent_context.task_context.append(task)
    ↓
Tick (Single Execution)
    ├── tick_world() → Update world_context
    ├── tick_tasks() → Decompose tasks
    └── tick_execution() → Call adapters
    ↓
Return State Snapshot
```

**Critical Gap**: No mechanism to continue ticking until task completion or failure.

### State Context Structure

**AgentContext** (`/home/user/Geist/agents/agent_context.py`):
```python
class AgentContext:
    world_context: List[str]         # World facts
    task_context: List[str]          # Tasks to complete
    execution_context: List[str]     # Execution steps
    execution_classes: List[Any]     # Available adapters
    function_log: List[str]          # Function call history
    subprocess_id: int               # For subprocess execution
    settings: AgentSettings          # Agent configuration
```

**Observation**: Context is append-only with no completion tracking or success criteria.

---

## Stopping Condition Patterns in Modern SDKs

### 1. Claude SDK (Anthropic) Pattern

The Anthropic Claude SDK uses an **agentic loop** with explicit stopping conditions:

```python
# Conceptual Claude SDK Pattern
class AgentLoop:
    def run(
        self,
        task: str,
        max_iterations: int = 25,
        max_tokens: int = 100000,
        timeout_seconds: Optional[float] = None
    ) -> AgentResult:
        """
        Run agent loop with stopping conditions:
        - Max iterations reached
        - Task marked complete by agent
        - Token budget exhausted
        - Timeout exceeded
        - Error threshold exceeded
        """
```

**Stopping Conditions**:
1. **Max Iterations**: Hard limit on reasoning cycles
2. **Task Completion**: Agent explicitly signals completion
3. **Token Budget**: Total tokens consumed across all API calls
4. **Timeout**: Wall-clock time limit
5. **Error Rate**: Consecutive failures threshold

**Key Features**:
- Agent can signal completion via special markers or tool calls
- Built-in conversation turn management
- Automatic context window management
- Tool use loop with automatic continuation

### 2. OpenAI Assistants API Pattern

OpenAI's Assistants API uses **runs** with status tracking:

```python
# OpenAI Pattern
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Run states: queued, in_progress, completed, failed, cancelled, expired
while run.status in ['queued', 'in_progress']:
    run = client.beta.threads.runs.retrieve(thread_id, run_id)
    time.sleep(1)
```

**Stopping Conditions**:
1. **Status Transitions**: `completed`, `failed`, `cancelled`, `expired`
2. **Function Call Resolution**: Waits for tool outputs
3. **Required Action**: Stops when user input needed
4. **Token Limits**: Automatic truncation or failure

### 3. LangChain Agent Pattern

LangChain uses **AgentExecutor** with explicit iteration controls:

```python
# LangChain Pattern
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=15,
    max_execution_time=60.0,
    early_stopping_method="generate",  # or "force"
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

result = executor.invoke({"input": "Task description"})
```

**Stopping Conditions**:
1. **max_iterations**: Maximum reasoning steps
2. **max_execution_time**: Timeout in seconds
3. **early_stopping_method**: How to handle max iterations
   - `generate`: Let agent generate final answer
   - `force`: Return current state
4. **Agent Finish Action**: Agent returns `AgentFinish` instead of `AgentAction`

---

## Proposed Architecture

### Design Principles

1. **Backward Compatibility**: Existing single-tick API remains functional
2. **Opt-in Loop Mode**: New endpoint for multi-tick execution
3. **Configurable Stopping**: Multiple condition types
4. **Observable Progress**: Streaming updates and state snapshots
5. **Graceful Degradation**: Partial results on early termination

### New Components

#### 1. AgentStoppingConditions

```python
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

class StoppingReason(Enum):
    """Why the agent stopped executing."""
    TASK_COMPLETE = "task_complete"
    MAX_ITERATIONS = "max_iterations"
    MAX_TOKENS = "max_tokens"
    TIMEOUT = "timeout"
    ERROR_THRESHOLD = "error_threshold"
    USER_INTERRUPT = "user_interrupt"
    NO_PROGRESS = "no_progress"
    CUSTOM_CONDITION = "custom_condition"

@dataclass
class AgentStoppingConditions:
    """Configuration for agent execution stopping conditions."""

    # Iteration limits
    max_iterations: Optional[int] = 25

    # Token budget (cumulative across all API calls)
    max_tokens: Optional[int] = None

    # Wall-clock timeout
    timeout_seconds: Optional[float] = None

    # Error handling
    max_consecutive_errors: int = 3
    max_total_errors: int = 10

    # Progress detection
    require_progress_every_n_ticks: Optional[int] = 5

    # Custom stopping function
    custom_condition: Optional[Callable[[AgentContext], bool]] = None

    # Completion markers
    completion_markers: List[str] = ["TASK_COMPLETE", "[DONE]", "✓"]

    # Task queue behavior
    stop_when_task_queue_empty: bool = True

    # Token efficiency
    warn_at_token_percentage: float = 0.8
```

#### 2. AgentExecutionResult

```python
@dataclass
class AgentExecutionResult:
    """Result of multi-tick agent execution."""

    # Completion status
    stopping_reason: StoppingReason
    success: bool

    # Execution metrics
    iterations_completed: int
    tokens_used: int
    execution_time_seconds: float

    # Final state
    final_context: AgentContext

    # Execution trace
    tick_history: List[Dict[str, Any]]
    function_calls: List[Dict[str, Any]]
    errors: List[Exception]

    # Output
    result: Optional[Any] = None
    partial_results: List[Any] = []

    # Warnings
    warnings: List[str] = []
```

#### 3. AgentExecutor

```python
class AgentExecutor:
    """
    Executes agent with stopping conditions.

    Wraps BaseAgent to provide multi-tick execution loop
    with configurable stopping conditions.
    """

    def __init__(
        self,
        agent: BaseAgent,
        stopping_conditions: Optional[AgentStoppingConditions] = None
    ):
        self.agent = agent
        self.conditions = stopping_conditions or AgentStoppingConditions()

        # Execution state
        self.iteration_count = 0
        self.token_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.start_time = None
        self.tick_history = []
        self.last_progress_tick = 0

    def execute(
        self,
        task: Optional[str] = None,
        stream_updates: bool = False
    ) -> AgentExecutionResult:
        """
        Execute agent until stopping condition met.

        Args:
            task: Initial task (if not already initialized)
            stream_updates: Yield intermediate results

        Returns:
            AgentExecutionResult with completion details
        """
        # Initialize
        if task:
            self.agent.initialize(task)
        self.start_time = time.time()

        # Execution loop
        while True:
            # Check stopping conditions BEFORE tick
            stop_check = self._check_stopping_conditions()
            if stop_check:
                return self._build_result(stop_check)

            # Execute one tick
            try:
                tick_result = self._execute_tick()
                self.tick_history.append(tick_result)
                self.consecutive_errors = 0

                # Update metrics
                self.iteration_count += 1
                self.token_count += tick_result.get('tokens_used', 0)

                # Check for completion markers
                if self._check_completion_markers(tick_result):
                    return self._build_result(StoppingReason.TASK_COMPLETE)

                # Stream update if requested
                if stream_updates:
                    yield self._build_partial_result()

            except Exception as e:
                self.error_count += 1
                self.consecutive_errors += 1
                self._handle_error(e)

                # Check error thresholds
                if self.consecutive_errors >= self.conditions.max_consecutive_errors:
                    return self._build_result(StoppingReason.ERROR_THRESHOLD)

    def _check_stopping_conditions(self) -> Optional[StoppingReason]:
        """Check all stopping conditions."""

        # Max iterations
        if (self.conditions.max_iterations and
            self.iteration_count >= self.conditions.max_iterations):
            return StoppingReason.MAX_ITERATIONS

        # Token budget
        if (self.conditions.max_tokens and
            self.token_count >= self.conditions.max_tokens):
            return StoppingReason.MAX_TOKENS

        # Timeout
        if self.conditions.timeout_seconds:
            elapsed = time.time() - self.start_time
            if elapsed >= self.conditions.timeout_seconds:
                return StoppingReason.TIMEOUT

        # Empty task queue
        if (self.conditions.stop_when_task_queue_empty and
            not self.agent._agent_context.task_context):
            return StoppingReason.TASK_COMPLETE

        # No progress detection
        if self.conditions.require_progress_every_n_ticks:
            if (self.iteration_count - self.last_progress_tick >=
                self.conditions.require_progress_every_n_ticks):
                if not self._has_made_progress():
                    return StoppingReason.NO_PROGRESS

        # Custom condition
        if self.conditions.custom_condition:
            if self.conditions.custom_condition(self.agent._agent_context):
                return StoppingReason.CUSTOM_CONDITION

        return None

    def _check_completion_markers(self, tick_result: Dict) -> bool:
        """Check if agent signaled completion."""
        execution_context = tick_result.get('execution_context', [])

        for marker in self.conditions.completion_markers:
            for item in execution_context:
                if marker in str(item):
                    return True
        return False

    def _has_made_progress(self) -> bool:
        """Detect if agent is making progress."""
        if len(self.tick_history) < 2:
            return True

        current = self.tick_history[-1]
        previous = self.tick_history[-2]

        # Check if context has changed
        current_state = (
            str(current.get('world_context', [])) +
            str(current.get('task_context', [])) +
            str(current.get('execution_context', []))
        )

        previous_state = (
            str(previous.get('world_context', [])) +
            str(previous.get('task_context', [])) +
            str(previous.get('execution_context', []))
        )

        if current_state == previous_state:
            return False

        # Update last progress tick
        self.last_progress_tick = self.iteration_count
        return True
```

### Modified AgentContext

Add execution tracking fields:

```python
class AgentContext:
    # ... existing fields ...

    # NEW: Execution tracking
    is_complete: bool = False
    completion_message: Optional[str] = None
    total_tokens_used: int = 0
    execution_start_time: Optional[float] = None
```

---

## Implementation Comparison

### Feature Matrix

| Feature | Geist (Current) | Geist (Proposed) | Claude SDK | OpenAI Assistants | LangChain |
|---------|-----------------|------------------|------------|-------------------|-----------|
| **Max Iterations** | ❌ | ✅ | ✅ | ❌ (implicit) | ✅ |
| **Token Budget** | ❌ | ✅ | ✅ | ✅ (implicit) | ❌ |
| **Timeout** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Task Completion Signal** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Error Thresholds** | Partial (3 retries) | ✅ | ✅ | ✅ | ✅ |
| **Progress Detection** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Custom Conditions** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Streaming Updates** | ❌ | ✅ | ✅ | ✅ (polling) | ✅ |
| **Intermediate Steps** | Partial | ✅ | ✅ | ✅ | ✅ |
| **State Persistence** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Subprocess Support** | ✅ | ✅ | ❌ | ✅ | ❌ |

### Interface Comparison

#### Current Geist API

```python
# Single-tick execution
POST /agent/initialize_task_and_tick
{
    "task": "Send email to user@example.com",
    "agent_settings": {...}
}

# Response: State after one tick
{
    "world_context": [...],
    "task_context": [...],
    "execution_context": [...]
}
```

#### Proposed Geist API

```python
# Multi-tick execution with stopping conditions
POST /agent/execute
{
    "task": "Send email to user@example.com",
    "agent_settings": {...},
    "stopping_conditions": {
        "max_iterations": 10,
        "max_tokens": 50000,
        "timeout_seconds": 60.0,
        "stop_when_task_queue_empty": true
    },
    "stream": false
}

# Response: Execution result
{
    "stopping_reason": "task_complete",
    "success": true,
    "iterations_completed": 3,
    "tokens_used": 1247,
    "execution_time_seconds": 4.2,
    "result": {...},
    "tick_history": [...],
    "warnings": []
}
```

#### Claude SDK Pattern (Conceptual)

```python
from anthropic import Anthropic

client = Anthropic()

# Agent loop with tools
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    tools=[...],
    messages=[{"role": "user", "content": "Task"}]
)

# Manual loop
while response.stop_reason == "tool_use":
    # Process tool calls
    tool_results = execute_tools(response.content)

    # Continue conversation
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        tools=[...],
        messages=[
            *conversation_history,
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results}
        ]
    )
```

**Key Difference**: Claude SDK requires manual loop implementation, whereas Geist can provide built-in executor.

#### OpenAI Assistants Pattern

```python
import openai

# Create run
run = openai.beta.threads.runs.create(
    thread_id="thread_abc123",
    assistant_id="asst_abc123"
)

# Poll until complete
while run.status in ['queued', 'in_progress', 'requires_action']:
    if run.status == 'requires_action':
        # Submit tool outputs
        run = openai.beta.threads.runs.submit_tool_outputs(...)
    else:
        time.sleep(1)
        run = openai.beta.threads.runs.retrieve(...)
```

**Key Difference**: OpenAI uses polling with status checks, whereas Geist uses synchronous execution.

---

## Migration Path

### Phase 1: Add Stopping Condition Infrastructure

**Files to Modify**:
- `agents/agent_context.py` - Add execution tracking fields
- `agents/stopping_conditions.py` - **NEW** - StoppingConditions dataclass
- `agents/agent_executor.py` - **NEW** - AgentExecutor class
- `agents/agent_result.py` - **NEW** - AgentExecutionResult dataclass

**Backward Compatibility**: Existing agents unchanged.

### Phase 2: Update API Endpoints

**Files to Modify**:
- `app/api/v1/endpoints/agent.py` (or equivalent)

**New Endpoints**:
```python
@router.post("/agent/execute")
async def execute_agent_with_loop(
    request: AgentExecuteRequest
) -> AgentExecutionResult:
    """Execute agent with stopping conditions."""

@router.post("/agent/execute/stream")
async def execute_agent_stream(
    request: AgentExecuteRequest
) -> StreamingResponse:
    """Execute agent with streaming updates."""
```

**Preserved Endpoints**:
- `/agent/initialize_task_and_tick` - Single tick (unchanged)
- `/agent/complete_text` - Direct completion (unchanged)

### Phase 3: Enhanced Context Management

**Improvements**:
1. Add `is_complete` flag to AgentContext
2. Track token usage per tick
3. Add progress detection heuristics
4. Implement completion markers

### Phase 4: Testing & Validation

**Test Cases**:
1. ✅ Max iterations reached
2. ✅ Task completion (empty queue)
3. ✅ Token budget exhausted
4. ✅ Timeout exceeded
5. ✅ Error threshold triggered
6. ✅ No progress detection
7. ✅ Custom stopping condition
8. ✅ Graceful degradation

---

## Code Examples

### Example 1: Basic Execution with Max Iterations

```python
from agents.gpt4_agent import GPT4Agent
from agents.agent_context import AgentContext
from agents.agent_settings import AgentSettings
from agents.agent_executor import AgentExecutor
from agents.stopping_conditions import AgentStoppingConditions

# Setup
settings = AgentSettings(max_tokens=1000, temperature=0.7)
context = AgentContext(settings=settings)
agent = GPT4Agent(api_key=os.getenv("OPENAI_API_KEY"), agent_context=context)

# Configure stopping conditions
conditions = AgentStoppingConditions(
    max_iterations=10,
    stop_when_task_queue_empty=True
)

# Execute
executor = AgentExecutor(agent, conditions)
result = executor.execute(task="Research Python async patterns")

# Result
print(f"Stopped: {result.stopping_reason}")
print(f"Iterations: {result.iterations_completed}")
print(f"Success: {result.success}")
```

### Example 2: Token Budget and Timeout

```python
conditions = AgentStoppingConditions(
    max_iterations=50,
    max_tokens=100000,  # 100k token budget
    timeout_seconds=120.0,  # 2 minute timeout
    warn_at_token_percentage=0.8
)

executor = AgentExecutor(agent, conditions)
result = executor.execute(task="Generate comprehensive report")

# Check warnings
if result.warnings:
    print("Warnings:", result.warnings)
```

### Example 3: Custom Stopping Condition

```python
def stop_on_email_sent(context: AgentContext) -> bool:
    """Stop when email adapter has been called."""
    for log_entry in context.function_log:
        if "SendgridAdapter" in log_entry and "send_email" in log_entry:
            return True
    return False

conditions = AgentStoppingConditions(
    max_iterations=20,
    custom_condition=stop_on_email_sent
)

executor = AgentExecutor(agent, conditions)
result = executor.execute(task="Send summary email to team")
```

### Example 4: Streaming Execution

```python
conditions = AgentStoppingConditions(max_iterations=15)
executor = AgentExecutor(agent, conditions)

# Stream updates
for partial in executor.execute(task="Long running task", stream_updates=True):
    print(f"Iteration {partial.iterations_completed}: {partial.current_state}")

# Final result
final = partial  # Last yielded value
```

### Example 5: Error Handling

```python
conditions = AgentStoppingConditions(
    max_iterations=25,
    max_consecutive_errors=3,
    max_total_errors=10
)

executor = AgentExecutor(agent, conditions)
result = executor.execute(task="Complex multi-step task")

if result.stopping_reason == StoppingReason.ERROR_THRESHOLD:
    print(f"Failed after {len(result.errors)} errors")
    for error in result.errors:
        print(f"  - {error}")
else:
    print(f"Completed successfully: {result.result}")
```

---

## Recommendations

### 1. **Adopt Multi-Condition Approach**

Implement multiple stopping conditions similar to LangChain's AgentExecutor:
- ✅ Max iterations (prevent infinite loops)
- ✅ Token budget (cost control)
- ✅ Timeout (wall-clock limit)
- ✅ Task completion signal (explicit success)
- ✅ Error thresholds (failure detection)
- ✅ Progress detection (stuck detection)

### 2. **Add Completion Signaling**

Allow agents to explicitly signal completion:
```python
# In execution_context
["Log progress: Starting research",
 "Search web for information",
 "TASK_COMPLETE: Research finished"]
```

### 3. **Implement Token Tracking**

Track cumulative token usage across all API calls:
```python
class AgentContext:
    total_tokens_used: int = 0

    def add_completion_tokens(self, completion):
        if hasattr(completion, 'usage'):
            self.total_tokens_used += completion.usage.total_tokens
```

### 4. **Progress Detection**

Detect when agent is "stuck" (no context changes):
```python
def _has_made_progress(self) -> bool:
    # Compare last N ticks for context changes
    # Return False if identical
```

### 5. **Streaming Support**

Add SSE (Server-Sent Events) endpoint for real-time updates:
```python
@router.post("/agent/execute/stream")
async def stream_execution():
    async def event_generator():
        for update in executor.execute(stream_updates=True):
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### 6. **Observability**

Add comprehensive execution traces:
```python
class AgentExecutionResult:
    tick_history: List[TickSnapshot]
    function_calls: List[FunctionCall]
    token_usage_timeline: List[int]
    execution_timeline: List[float]
```

---

## Comparison Summary

### Geist (Current) vs. Claude SDK

| Aspect | Geist | Claude SDK |
|--------|-------|------------|
| **Execution** | Single tick, manual | Manual loop, user-controlled |
| **Tools** | JSON adapters, dynamic | Built-in tools, type-safe |
| **Stopping** | None | User implements loop logic |
| **State** | Context lists | Message history |
| **Flexibility** | High (custom adapters) | Medium (predefined tools) |

**Verdict**: Geist can surpass Claude SDK by adding built-in execution loop.

### Geist (Current) vs. OpenAI Assistants

| Aspect | Geist | OpenAI Assistants |
|--------|-------|-------------------|
| **Execution** | Synchronous tick | Asynchronous run |
| **State** | In-memory context | Server-side threads |
| **Tools** | Dynamic adapters | Predefined functions |
| **Stopping** | None | Status-based |
| **Control** | Full (self-hosted) | Limited (API-based) |

**Verdict**: Geist offers more control; should add async execution support.

### Geist (Proposed) vs. LangChain

| Aspect | Geist (Proposed) | LangChain |
|--------|------------------|-----------|
| **Execution** | AgentExecutor | AgentExecutor |
| **Stopping** | 7+ conditions | 3 conditions |
| **Progress** | Built-in detection | None |
| **Observability** | Full traces | Limited |
| **Flexibility** | High | Very High |

**Verdict**: Proposed Geist matches or exceeds LangChain capabilities.

---

## Conclusion

The current Geist BaseAgent architecture is **well-designed but incomplete** for production agent systems. By adding:

1. **AgentExecutor** - Multi-tick execution loop
2. **AgentStoppingConditions** - Configurable termination criteria
3. **AgentExecutionResult** - Rich result objects with traces
4. **Token & Progress Tracking** - Resource and progress monitoring

Geist can match or exceed the capabilities of Claude SDK, OpenAI Assistants, and LangChain while maintaining its unique advantages:

- ✅ Flexible adapter system
- ✅ Three-phase reasoning (world/task/execution)
- ✅ Self-hosted control
- ✅ Subprocess support
- ✅ Rich context management

**Next Steps**:
1. Implement `AgentStoppingConditions` dataclass
2. Create `AgentExecutor` wrapper class
3. Add token tracking to AgentContext
4. Create new API endpoint `/agent/execute`
5. Write comprehensive tests
6. Document migration guide

This enhancement will position Geist as a **production-ready agent framework** competitive with major SDK offerings while preserving its unique architectural advantages.

---

## References

- **Geist BaseAgent**: `/home/user/Geist/agents/base_agent.py`
- **Geist AgentContext**: `/home/user/Geist/agents/agent_context.py`
- **Geist OnlineAgent**: `/home/user/Geist/agents/online_agent.py`
- **Anthropic Claude SDK**: https://docs.anthropic.com/en/docs/agents
- **OpenAI Assistants API**: https://platform.openai.com/docs/assistants
- **LangChain AgentExecutor**: https://python.langchain.com/docs/modules/agents/

---

**Report Generated**: 2026-01-04
**Branch**: `claude/agent-stopping-conditions-MuCyp`
**Author**: Claude (Anthropic AI)
