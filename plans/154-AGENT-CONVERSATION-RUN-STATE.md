# Agent, Conversation, and Run State

## Goal

Use one orchestration path for streaming and non-streaming chat while keeping
agent-wide state, conversation history, and per-run execution state explicitly
scoped. Preserve structured message roles through local runners so persisted
turns become reliable model context.

## Design

1. Keep `AgentContext` limited to agent identity, settings, tools, and durable
   world/task/execution state.
2. Add `ConversationState` for one chat session. It owns the structured message
   view hydrated from persisted history and the persisted chat identifier.
3. Add `AgentRunContext` for one invocation. It owns the run identifier,
   transcript, tool trajectory, artifacts, status, cancellation state, and
   exactly-once terminal persistence bookkeeping.
4. Keep persistence in the existing SQLAlchemy/helper layer. The state objects
   provide scoped data and transition methods; no repository abstraction is
   introduced.
5. Route both streaming and non-streaming modern completions through
   `ChatOrchestrator`. Retain legacy agents behind the existing compatibility
   fallback until their model-turn contract is migrated.
6. Add a structured runner message method. MLX and Transformers runners must
   apply their tokenizer chat templates to the original role-preserving message
   sequence instead of flattening history into one user prompt.

## Tests

- Unit tests for `ConversationState` hydration and `AgentRunContext` transition
  snapshots.
- Orchestrator tests proving structured history reaches the backend and one
  terminal turn is persisted.
- Local runner tests proving user/assistant roles reach MLX and Transformers
  chat templates without flattening.
- Route tests proving streaming and non-streaming modern completions share the
  orchestrator.
- Playwright coverage proving a created conversation persists through reload
  and a follow-up request uses the returned session ID.
- Native MLX browser smoke proving a follow-up can recall an exact earlier user
  message.

## Validation

- Ruff, formatting, and mypy for changed Python.
- Focused state, orchestrator, route, and runner tests.
- Full backend and frontend test suites when feasible.
- Frontend production build and Playwright browser suite.
- Python/frontend dependency audits if dependency manifests change.
- Native `make run MLX_BACKEND=1` and live browser multi-turn chat.
