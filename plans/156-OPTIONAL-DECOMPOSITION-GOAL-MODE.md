# Adaptive Agentic Harness

## Scope

One default-on Agentic Mode setting enables a model-directed, budgeted tool loop.
There is no intent router, mandatory preplanner, independent judge, or worker queue.
Backends without native tool calling retain their single-turn behavior.

## Model and harness responsibilities

The executor sees the conversation and available tools immediately. It can inspect
the workspace, create or revise a visible plan with `agent.plan.update`, and proceed
without a plan for simple answers. Plans are working state, not a prescribed queue.

One loop makes at most 48 model calls per run (configurable internally up to 200),
with at most 10 tools in one model response. Every model call spends budget,
including consecutive tool-use responses. The existing direct-mode limits remain.
Tool output has a rolling context budget that prioritizes new observations while
retaining call/result IDs and complete protocol sequences.

The executor stops with `agent.goal.complete` or yields with `agent.goal.wait`.
Ordinary prose is not completion. Exhaustion yields `budget_limited`; cancellation
yields `paused`. Neither claims success. Resuming extends the model-call budget
while preserving the goal ID, objective, plan, observations, and workspace identity.

Completion requires nonblank evidence, no open tasks, explanations for skipped
tasks, and recorded observation references after tool work. All-skipped plans and
invented references are rejected. References make the model's claim traceable;
they are not semantic proof that the requested feature is correct.

## Persistence and live instructions

`GoalRuntime` owns synchronized working state. `GoalStore` persists a checkpoint
containing the transcript, plan, observations, instruction inbox, and workspace ID.
The database migration adds `AgentGoal.checkpoint_json`. User/chat-scoped lookup
resumes unfinished goals when the user sends another message.

`ChatOrchestrator` owns model/tool sequencing, permissions, SSE, cancellation, and
terminal chat history persistence. Checkpoints retain complete tool-result blocks;
a mid-tool interruption must be reconciled against the workspace before retrying.
This is not an exactly-once side-effect transaction. A new chat is attached when
its first terminal history record is saved; crash recovery before that attachment
is not supported by chat lookup.

`POST /agent/runs/{run_id}/instructions` accepts owner-scoped, idempotent instruction
IDs with bounded text and queue size. The active composer submits here without
aborting the stream. Accepted instructions are checkpointed before acknowledgment.
They interrupt approval waits and supersede unstarted calls from stale output.
Already-running tools are not rolled back. Delivery occurs only between complete
assistant/tool-result blocks, never in the middle of a provider tool sequence.
The inbox closes atomically at terminal/budget boundaries; late accepted input
at the budget remains queued for the next run. Duplicate active runs for an
existing chat are rejected within the current process.

## Tool boundaries

Each model tool call becomes a fresh server-issued invocation with normalized
arguments. Approval is bound to its identity, tool name, and exact arguments,
and consumed once. Duplicate provider call IDs are rejected. Existing per-call
terminal gates, unattended denial, tool validation, and sandbox hardening remain.

`CodingWorkspace` routes file and terminal operations to the same configured
root or Docker/Podman session, including new chats. Conflicting explicit roots
fail closed. File path containment, secret-file restrictions, and write approvals
remain. Host-reaching/networked terminal calls still require fresh approval.

Chat coding tools always share a container session; `GEIST_EXEC_PERSISTENT`
remains an opt-in for legacy clients. The existing session TTL setting applies.
Unconfigured container files live in tmpfs for that container's lifetime, not in
durable storage. An explicit workspace mount is required to retain files across
container expiry/recreation. A resumed agent should inspect before trusting old
observations. Native local execution retains its existing platform-dependent
isolation; this refactor does not make an unsandboxed platform sandboxed.

## Verification

- Long consecutive tool sequences and exact model-call budget accounting.
- Explicit completion, wait/resume, cancellation, and direct-mode contracts.
- Durable checkpoint round trips and user/chat ownership.
- Steering during generation, approvals, pre-dispatch, and completion; budget-edge
  instruction retention, idempotence, and provider-valid transcripts.
- Single-use approval and mismatched/repeated invocation rejection.
- File/terminal interoperability in an actual isolated container.
- Focused frontend hook, transcript, settings tests and production build.
- Containerized backend tests and Docker/browser smoke per the Geist test loop.
