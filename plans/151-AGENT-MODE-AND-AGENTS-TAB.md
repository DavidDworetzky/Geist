# Plan 151: Agent Mode & Agents Tab

## Research Summary

### Current State of Geist Agents

Geist already has a tick-based agent architecture (`agents/base_agent.py`) with three execution phases per tick:

1. **tick_world()** - Executive-level world reasoning (optional)
2. **tick_tasks()** - Decompose the main task into `|`-delimited subtasks
3. **tick_execution()** - Generate a JSON function call against available adapters

State is tracked in `AgentContext` (`agents/agent_context.py`) with three context lists: `world_context`, `task_context`, `execution_context`. Agents persist to the `agent` table via `_save()`. The adapter system (`adapters/adapter_registry.py`) dynamically discovers tools.

**Key gap**: The tick system exists but is never driven to completion autonomously. There is no loop that calls `tick()` repeatedly, evaluates whether the task is done, and stops. The `initialize()` method can spawn a subprocess (`tick.py`), but there is no completion detection, no verification, and no feedback loop from tool execution results back into the next tick.

The existing **Workflow system** (`app/services/workflow_execution.py`, `WorkflowBuilder.tsx`) provides a DAG of steps with ReactFlow visualization, but it is a static pipeline — steps are defined ahead of time, not generated dynamically by an LLM.

### What OpenCode Does

OpenCode (Go, Bubble Tea TUI) uses a **flat iterative tool-use loop** (ReAct-style):

```
loop:
  1. Stream LLM response
  2. If finish_reason == "tool_use": execute tools, append results, GOTO 1
  3. If finish_reason == "end_turn": return
```

Key design decisions:
- **No explicit planner or DAG** — the LLM implicitly plans via conversation history
- **No post-execution verifier** — validation is pre-execution (permission gates, read-before-edit enforcement, safe command allowlists)
- **Two-level agent hierarchy**: a `coder` agent with full tools can spawn read-only `task` sub-agents for research
- **No dedicated agents tab** — agent execution renders inline in chat with spinners and status indicators
- **Auto-compaction** at 95% context usage via a summarizer agent
- **Context-based cancellation** — Go context propagation for clean teardown

### Analysis: Stack vs DAG vs Flat Loop

| Model | Pros | Cons | When to use |
|-------|------|------|-------------|
| **Flat loop** (OpenCode) | Simple, LLM handles planning, minimal code | No parallelism, no explicit progress tracking, hard to resume | Single-agent chat-like tasks |
| **Stack** (current Geist tick) | Natural for decompose-then-execute, can push/pop subtasks | Sequential only, no parallelism, awkward for independent subtasks | Hierarchical task decomposition |
| **DAG** | Parallel execution, explicit dependencies, visual progress | Complex to implement, LLM must generate valid DAG structure, overkill for most tasks | Multi-agent workflows with independent branches |

**Recommendation**: Use a **hybrid approach**. The primary agent mode should be a **ReAct-style tool-use loop** (like OpenCode) because it is battle-tested and maps directly onto how LLM APIs work with tool calling. Layer on a **task stack** for the "tick" decomposition that Geist already has, but make it optional — used when the agent decides to break a problem into phases. Reserve the DAG for the existing Workflow system, which handles user-defined static pipelines.

### Analysis: Do We Need a Verifier?

**Yes, but not as a separate post-hoc pass.** Analysis:

1. **OpenCode's approach** (pre-execution gates only) works because it targets developers who watch the output and can interrupt. Geist's "give it a task and walk away" mode needs more.

2. **A verifier agent** should run after the primary agent declares completion. It reviews what was done and can either accept the result or send the agent back for corrections. This is cheap (one extra LLM call) and catches the most common failure mode: the agent *thinks* it's done but missed something.

3. **Implementation**: A lightweight "judge" — a separate LLM call with a focused prompt that receives the task description, the sequence of actions taken, and the final state. Returns `{done: true}` or `{done: false, reason: "..."}`. If not done, the reason is appended to context and the agent loop continues.

This is analogous to the "critic" pattern in multi-agent systems but kept minimal — one verifier, not a committee.

---

## Proposed Architecture

### Core Concept: AgentTask

An `AgentTask` is a top-level unit of autonomous work. It wraps a goal, an agent, and a run loop.

```
AgentTask
├── task_id (UUID)
├── goal (str) — what the user wants done
├── status (pending | running | verifying | completed | failed | cancelled)
├── agent_config — which model, provider, tools, settings
├── steps[] — ordered list of AgentStep records
├── created_at, updated_at
└── result — final output/summary
```

### Core Concept: AgentStep

Each iteration of the agent loop produces a step:

```
AgentStep
├── step_id (UUID)
├── task_id (FK)
├── step_number (int)
├── type (reasoning | tool_call | tool_result | verification)
├── content — LLM output text, tool call JSON, or tool result
├── tokens_used
├── created_at
└── duration_ms
```

### Agent Run Loop (Backend)

```python
class AgentRunner:
    """Drives an agent task to completion."""

    def run(self, task: AgentTask) -> AgentTask:
        task.status = "running"
        max_iterations = task.max_iterations or 50

        for i in range(max_iterations):
            # 1. Build messages from task goal + step history
            messages = self._build_messages(task)

            # 2. Call LLM with tool definitions
            response = self.agent.complete_with_tools(messages)

            # 3. Record reasoning step
            self._record_step(task, "reasoning", response.text)

            # 4. If LLM returned tool calls, execute them
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = self._execute_tool(tool_call)
                    self._record_step(task, "tool_call", tool_call)
                    self._record_step(task, "tool_result", result)
                continue  # Loop back for next LLM turn

            # 5. If LLM says done (no tool calls), run verifier
            if self.verifier:
                verdict = self.verifier.verify(task)
                self._record_step(task, "verification", verdict)
                if not verdict.done:
                    # Append feedback and continue
                    self._record_step(task, "reasoning",
                        f"Verifier feedback: {verdict.reason}")
                    continue

            # 6. Task complete
            task.status = "completed"
            task.result = response.text
            return task

        task.status = "failed"
        task.result = "Max iterations reached"
        return task
```

### Verifier

```python
class AgentVerifier:
    """Lightweight judge that checks if a task is actually complete."""

    VERIFY_PROMPT = """You are a task completion verifier.
    Given the original goal and the actions taken, determine if the task
    is fully complete. Respond with JSON:
    {"done": true/false, "reason": "explanation if not done"}"""

    def verify(self, task: AgentTask) -> VerifyResult:
        summary = self._summarize_steps(task.steps)
        response = self.llm.complete(
            prompt=f"Goal: {task.goal}\n\nActions taken:\n{summary}",
            system_prompt=self.VERIFY_PROMPT
        )
        return VerifyResult.from_json(response)
```

### Tool Calling Integration

The existing adapter system needs to be bridged to the OpenAI/Anthropic tool-calling format:

1. **Convert adapters to tool definitions** — Each adapter method becomes a tool with a JSON schema for parameters. `find_adapter_classes()` already returns `(class_name, [method_names])`. Extend this to also return parameter schemas via inspection.

2. **Native tool calling** — Use the LLM provider's native tool/function calling API instead of prompting the LLM to emit JSON and parsing it. This is more reliable and is how OpenCode does it.

3. **Permission layer** (inspired by OpenCode) — Categorize tools as read-only (always allowed) vs. side-effecting (require approval or run in auto-approve mode for fully autonomous tasks).

---

## Agents Tab (Frontend)

### New Route: `/agents`

A new top-level tab in the navigation alongside Chat, Workflows, Files, Settings.

### Page Layout

```
┌─────────────────────────────────────────────────────────┐
│  [+ New Agent Task]                        [Filter ▾]   │
├──────────────┬──────────────────────────────────────────┤
│              │                                          │
│  Task List   │  Task Detail                             │
│              │                                          │
│  ● Running   │  Goal: "Refactor auth module to use..."  │
│    Task A    │  Status: ● Running (step 12/50)          │
│              │  Model: claude-3.5-sonnet                 │
│  ✓ Done      │  Started: 2 min ago                      │
│    Task B    │                                          │
│              │  ┌─ Step Timeline ──────────────────┐    │
│  ✗ Failed    │  │ 1. 🤔 Reasoning: "First I'll..." │    │
│    Task C    │  │ 2. 🔧 read_file(auth.py)         │    │
│              │  │ 3. 📄 Result: [file contents]     │    │
│  ○ Pending   │  │ 4. 🤔 Reasoning: "I see the..."  │    │
│    Task D    │  │ 5. 🔧 edit_file(auth.py, ...)     │    │
│              │  │ 6. 📄 Result: "OK"                │    │
│              │  │ 7. ✅ Verification: PASS           │    │
│              │  └──────────────────────────────────┘    │
│              │                                          │
│              │  [Cancel]  [Retry]  [View Result]        │
└──────────────┴──────────────────────────────────────────┘
```

### Key UI Components

1. **AgentTaskList** — Left sidebar showing all tasks with status badges, sorted by recency. Filterable by status.

2. **AgentTaskDetail** — Right panel showing:
   - Task metadata (goal, model, timing, token usage)
   - Live step timeline with collapsible entries
   - Streaming status indicator when running
   - Action buttons (cancel, retry, view result)

3. **NewAgentTaskDialog** — Modal for creating tasks:
   - Goal text area
   - Model/provider selector (reuse `AgentConfigSection`)
   - Tool selection (which adapters to enable)
   - Max iterations slider
   - Auto-approve toggle (skip permission prompts)

4. **StepTimelineEntry** — Individual step rendering:
   - Reasoning steps: show LLM text (collapsible if long)
   - Tool calls: show function name + params in a code block
   - Tool results: show output (truncated with expand)
   - Verification: show pass/fail badge with reason

### Real-Time Updates

Use SSE (Server-Sent Events) or WebSocket from the backend to push step updates to the frontend as the agent runs. The agent run loop emits events after each step:

```
event: agent_step
data: {"task_id": "...", "step": {...}}

event: agent_status
data: {"task_id": "...", "status": "verifying"}
```

The React frontend subscribes via `EventSource` and updates the step timeline in real time.

---

## Implementation Phases

### Phase 1: Agent Run Loop (Backend)

1. Create `app/models/database/agent_task.py` — `AgentTask` and `AgentStep` SQLAlchemy models
2. Create `agents/agent_runner.py` — The core run loop with tool execution
3. Bridge the adapter registry to produce OpenAI-format tool definitions
4. Add native tool calling support to `OnlineAgent` (parse tool_calls from response, not free-text JSON)
5. Create `agents/agent_verifier.py` — Lightweight completion checker
6. Add API endpoints:
   - `POST /agent/tasks` — Create and start a task
   - `GET /agent/tasks` — List tasks
   - `GET /agent/tasks/{id}` — Get task with steps
   - `POST /agent/tasks/{id}/cancel` — Cancel running task
   - `GET /agent/tasks/{id}/stream` — SSE stream of step events

### Phase 2: Agents Tab (Frontend)

1. Add `/agents` route and navigation entry in `App.tsx`
2. Build `AgentTaskList` component with status filtering
3. Build `AgentTaskDetail` with step timeline
4. Build `NewAgentTaskDialog` reusing `AgentConfigSection`
5. Implement SSE client hook (`useAgentTaskStream`) for live updates
6. Build `StepTimelineEntry` with type-specific rendering

### Phase 3: Verifier & Robustness

1. Integrate verifier into the run loop (opt-in via task config)
2. Add auto-compaction when conversation history gets large (inspired by OpenCode's 95% threshold)
3. Add graceful cancellation (propagate cancel to running tool calls)
4. Add retry logic for failed tasks (restart from last successful step)

### Phase 4: Permission System

1. Categorize adapters/tools as read-only vs. side-effecting
2. Add permission request/approval flow for side-effecting tools
3. In auto-approve mode, skip permission prompts (for fully autonomous tasks)
4. Log all tool executions for audit trail

---

## Why Not a Full DAG?

The Workflow system already handles DAGs for user-defined static pipelines. For autonomous agent mode, a DAG adds complexity without proportional value:

- The LLM doesn't reliably generate valid DAG structures
- Most agent tasks are inherently sequential (read -> think -> act -> verify)
- Parallelism is rarely needed within a single agent's reasoning loop
- If parallel sub-tasks are needed, the agent can spawn them (like OpenCode's `task` sub-agents)

The **task stack** from the existing tick system can optionally layer on top of the flat loop for cases where the agent wants to decompose a problem into phases, but the primary execution model should be the simpler ReAct loop.

## Why a Verifier?

Unlike OpenCode (which is developer-facing with a human watching), Geist's agent mode targets "give it a task and walk away." Without a verifier:

- The agent may hallucinate completion
- Partial work may be marked as done
- There is no feedback loop to catch obvious mistakes

The verifier is cheap (one LLM call), adds minimal latency, and catches the 80% case. It is not a full testing framework — it is a sanity check.

---

## Open Questions

1. **Sub-agent spawning** — Should the primary agent be able to spawn read-only research sub-agents (like OpenCode's `task` agent)? This is useful but adds complexity. Recommend deferring to Phase 5.

2. **Context window management** — Auto-compaction (summarizing old steps to free up context) is important for long-running tasks. What threshold should trigger it?

3. **Cost controls** — Should there be a token budget per task? A dollar limit? Both?

4. **Persistence of running tasks across server restarts** — The step history is in the DB, but resuming a mid-flight agent from persisted state requires careful design. Recommend starting with "restart from beginning" and adding resume later.

5. **Integration with Workflows** — An agent task could be a step in a workflow (the `AGENT` step type already exists but is mocked). Wire it up in Phase 5.
