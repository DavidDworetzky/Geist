# 157 — RLM Agent (minimal recursive language model layer)

## Motivation

Prime Intellect's Prime Agent (and the underlying `alexzhang13/rlm` library)
demonstrate a "code-first" agent execution model — the Recursive Language
Model (RLM):

- The model gets one capability: a persistent Python REPL.
- Supplemental context is **a variable in the REPL** (`context`), not text
  stuffed into the prompt window. The model slices/greps/summarizes it with
  code, so context size is bounded by process memory, not the model window.
- Tools are plain function calls in generated code, not a JSON tool-call
  protocol.
- Recursion is a function call: `rlm_query(...)` spawns a child agent with its
  own REPL and returns its final answer as a Python value.

This is a natural fit for Geist: `BaseAgent` already abstracts providers
(local + online), `AgentContext` already carries durable state and a
`ToolDispatcher`, and the adapter registry gives us a tool inventory. The RLM
layer is an *alternative execution surface* over that existing infrastructure —
notably it sidesteps the fragile parse-JSON-and-retry path that small local
models struggle with, since fenced Python is an easier emission format.

## Scope (minimal slice)

In scope:

- `agents/rlm_agent.py`: `RlmAgent` (wraps any `BaseAgent`) + `RlmEnvironment`
  (persistent in-process REPL) + `RlmResult`.
- REPL bindings: `context`, `llm_query`, `rlm_query`, `call_tool`,
  `list_tools`, `FINAL`.
- Depth-limited recursion; at the depth limit `rlm_query` degrades to a
  one-shot `llm_query` instead of failing.
- Output truncation before feeding execution results back to the model.
- Unit tests with a scripted fake agent (no network, no model weights).

Out of scope (deliberately, for this slice):

- Sandboxing. Code runs via `exec` in-process, like the reference library's
  default "local" environment. This is the same trust level as letting
  adapters run arbitrary effects; a subprocess/Docker REPL is the obvious
  next hardening step and the `RlmEnvironment` seam is where it plugs in.
- API routes / UI. `RlmAgent` is a library-level construct for now.
- Continual-harness features (skill authoring, /refine, scheduling).
- Async/background children (`job_queue` integration).

## Design

### Composition over inheritance

`RlmAgent` wraps a `BaseAgent` rather than subclassing it. Subclassing would
force reimplementing the four provider completion methods; wrapping means any
existing agent (LocalAgent with any runner, OnlineAgent with any provider)
gains RLM behavior unchanged, and heterogeneous recursion (online root, local
children) is a constructor argument away later.

### The loop protocol

Per iteration:

1. Render a prompt: the task, a note that `context` is loaded (with its size),
   and the code/output history so far.
2. `complete_text(...)` with a fixed RLM system prompt.
3. If the reply contains fenced Python, execute all fences in the persistent
   namespace, capture stdout (+ traceback on error), truncate, append to
   history.
4. If `FINAL(answer)` was called during execution, return that answer.
5. If the reply contains no code, treat the text itself as the final answer
   (lets the model answer trivial tasks in one turn and guarantees progress).
6. After `max_iterations` without `FINAL`, return `answer=None` with the
   transcript for inspection.

### REPL bindings

| name | behavior |
|---|---|
| `context` | the supplemental context object, verbatim (str or anything else) |
| `llm_query(prompt, system_prompt=None)` | one-shot completion on the wrapped agent, returns text |
| `rlm_query(prompt, context=None)` | run a child `RlmAgent` (fresh REPL, depth+1) to completion, return its answer; at `max_depth`, degrades to `llm_query` with the context inlined (truncated) |
| `call_tool("Adapter__action", **kwargs)` | dispatch through the existing `ToolDispatcher`; returns the result or raises so the model sees the error |
| `list_tools()` | qualified names + descriptions + parameters of the registered adapter actions |
| `FINAL(answer)` | end the session with this answer |

Errors inside generated code (including bad `call_tool` names) surface as
tracebacks in the execution output — the feedback channel the model already
watches — rather than crashing the loop.

## Testing

`tests/agents/test_rlm_agent.py` uses a `ScriptedAgent(BaseAgent)` returning
canned OpenAI-shaped completions. Covered: direct-answer path, persistent
namespace across iterations, `context` access, error feedback, recursion via
`rlm_query`, depth-limit degradation, `call_tool` through a real
`ToolDispatcher`, output truncation, and iteration exhaustion.

## Future work

- Subprocess/Docker `RlmEnvironment` for real isolation.
- `llm_query_batched` / background `rlm_query` via the existing job queue.
- Persist RLM transcripts through `AgentContext.snapshot`.
- Route-level exposure (chat endpoint flag) once hardened.

## References

- Prime Agent: https://github.com/PrimeIntellect-ai/prime-agent
- RLM inference library: https://github.com/alexzhang13/rlm
- pi (the harness Prime Agent builds on): https://github.com/earendil-works/pi
