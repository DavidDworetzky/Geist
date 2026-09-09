# Bounded malformed tool-call recovery

Introduce a shared, typed model-protocol error for the strict local tool parser.
The chat harness retries only that error, at most twice per run, counting failed
generations against the existing model-turn and goal budgets. Parsing must still
validate the entire turn before any calls are dispatched. Prior successful tool
results remain in context and are never replayed by the harness.

Close failed streams, retain already-visible prose, apply pending user instructions
before the next generation, and honor cancellation. Give the next generation a
format correction without exposing or retaining raw malformed argument payloads.
Exhaustion gets an actionable public error; unrelated backend exceptions are not
retried. Typed-parameter JSON failures retain a safe decoder reason and offset.

Verify malformed typed JSON, stream atomicity, recovery after earlier successful
tools, bounded exhaustion, goal budget accounting, cancellation, and non-protocol
exceptions with focused parser and orchestrator tests in Docker and native Python.
