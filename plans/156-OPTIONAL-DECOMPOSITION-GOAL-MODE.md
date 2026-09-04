# Agentic Mode Decomposition and Goal Loop

## Goal

Add one **Agentic mode** control to native Geist chat. When enabled with a
native tool-call-capable backend it always:

1. asks the selected model for a small structured task plan before execution;
2. keeps the same agent session working across multiple completed model/tool
   turns until the agent explicitly marks the objective complete or a bounded
   turn budget is reached.

The implementation follows Oh My Pi's core separation: todos describe the route,
the goal preserves the destination, and normal tool execution remains inside the
existing chat orchestrator. It deliberately does not add an independent LLM
judge or a multi-agent work queue in this slice.

## Contracts

### Setting and request

Add `user_settings.agentic_mode_enabled`, defaulting to `true`, and expose one
toggle at the top of General settings. Extend `CompleteTextParams` with:

- `agentic_mode: bool = True`

There is no intent router. The persisted setting is copied into each browser
request, while API callers get agentic execution by default unless they
explicitly opt out. The continuation budget is an internal bounded default.

### Plan

A decomposition pass always runs when Agentic mode is enabled. It uses the same selected backend without tools and requests
strict JSON containing at most twelve tasks. Each normalized task has:

- stable ID;
- concise title;
- acceptance criteria;
- status (`pending`, `in_progress`, `completed`, `blocked`, or `skipped`);
- optional evidence.

Malformed decomposition output falls back to one task representing the original
request, so enabling decomposition cannot prevent the real agent turn from
starting. The model receives the normalized plan in its execution context and an
`agent.plan.update` tool for recording progress.

Backends without native tool calling keep the existing single-turn behavior.
They cannot reliably emit the explicit plan and goal control tools, so the
harness must not spin until its budget or infer success from ordinary prose.

### Goal

One `AgentGoal` row records the objective, run/chat ownership, status, current
plan, turn budget, and turns used. The statuses are:

- `active`
- `complete`
- `paused`
- `budget_limited`
- `failed`

The model receives an `agent.goal.complete` tool whenever Agentic mode is active. Completion is
rejected while decomposed tasks remain pending, in progress, or blocked. The tool
therefore expresses the executor's completion claim while deterministic state
guards the claim.

After an ordinary assistant final response:

- non-agentic mode ends as today;
- Agentic mode continues with a hidden continuation message unless the completion
  tool succeeded;
- reaching `goal_max_turns` persists `budget_limited` and returns control to the
  user without claiming success;
- cancellation or an execution failure persists a resumable/diagnostic terminal
  status rather than completion.

### Streaming/UI

Add `plan` and `goal` SSE events. General settings exposes one default-on
**Agentic mode** toggle, and the active transcript renders the task checklist,
goal turn count, and terminal goal status. Existing tool approval UI continues
to gate every side-effecting tool call during every continuation turn.

## Service Boundaries

- `GoalRuntime` owns in-run synchronized goal/plan mutation and completion
  guards.
- `GoalStore` owns SQLAlchemy persistence and chat attachment.
- `TaskDecomposer` owns the bounded model pre-pass and normalization.
- `ChatOrchestrator` owns lifecycle composition, SSE ordering, continuation, and
  exactly-once chat persistence.
- The tool registry exposes orchestration tools only when the `ToolContext`
  enables Agentic mode.

## Safety and Bounds

- Goal continuation is capped at twenty turns by schema and defaults to eight.
- Existing per-turn model-round and tool-call limits remain in force.
- Decomposition has a twelve-task cap and bounded title/criteria/evidence text.
- The default coding tool set includes contained file listing, bounded reads,
  text search, exact edits, writes, and the configured terminal backend.
- Workspace paths cannot escape the configured root; common credential files
  are unavailable, and file writes/edits still require user approval.
- Terminal commands require a fresh approval when they can reach a host
  workspace or network. Auto-approve and standing grants cannot waive this
  gate, and the destructive-command hardline applies to every host-reaching
  execution backend.
- Plan and goal bookkeeping tools never require approval; they cannot produce
  user-impacting side effects.
- Orchestration tools are read/control-plane operations and never bypass the
  permission decision for actual side effects.
- Unattended routine runs keep their current deny posture for approval-gated
  tools.

## Non-goals

- No subagent spawning or durable Kanban-style worker queue.
- No intent classifier or automatic per-request mode selection.
- No separate completion-judge model.
- No plan-approval gate; tool approvals remain the authorization boundary.
- No migration of the legacy `BaseAgent.tick()` OODA endpoint.

## Tests

- Decomposer JSON normalization and single-task fallback.
- Goal store lifecycle and ownership.
- Orchestration tool availability, plan updates, and guarded completion.
- Direct mode behavior remains unchanged.
- Agentic mode automatically continues, completes explicitly, and stops at budget.
- Approval/deny/cancel behavior remains correct inside goal continuation.
- SSE reducer and UI tests cover toggles, plan progress, and goal status.
- Docker/backend and frontend smoke checks per the Geist test loop.
