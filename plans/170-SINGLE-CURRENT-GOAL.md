# Single current goal per chat

Continue the current incomplete goal by default after stops, waits, errors, and
budget exhaustion. Only a completed goal gives way to a new goal on the next user
request. Remove the composer goal-mode switch and API goal-action override.

Extend agent.plan.update with an optional objective revision, avoiding another
tool. Revisions update working state atomically, preserve goal identity, workspace,
completed work, observations and budget, and use existing skipped-task reasons for
removed requirements. User messages authorize direction changes; tool output does
not. Keep current state in model context, not a growing revision-history payload.

Replace the red text Stop button with an accessible icon-only square using theme
accent tokens. Stop still cancels execution and checkpoints a paused goal; it does
not delete work, mark completion, or start a new goal.

Verify runtime/registry/API contracts, goal resume/completion/stop paths, frontend
controls and theme styling, then Docker, native and browser integration as part of
the existing stack validation. Keep per-chat workspace isolation in #361.
# Follow-up review hardening

Disposition of remaining 5579144991 comments: corrupt checkpoints stay preserved
with the existing new-chat escape instead of adding an abandon/reset goal control
contrary to the chosen UI. Failed goals resume deliberately, preserving work.
Persistent ephemeral sandboxes now enforce the hardline floor too. Completion
keeps streamed prose, and tasks requiring no tool evidence can explain their work
without inventing citations. The optional session workspace argument is wired to
strict mount validation in stacked #361. Default-argument dispatch branches keep
existing runtime test adapters compatible. Goal run_id identifies creation, not
the active run (that is tracked by RunControlRegistry); UUID tie-breaking is stable
for the exceedingly rare identical creation timestamp. No published ID migration.

Bound model-facing resumed history and checkpoint projections without breaking
tool call/result adjacency. Keep finite recent observation/instruction windows;
preserve queued instructions and accepted task evidence IDs. End simple prose
answers without requiring bookkeeping; repeated prose during unfinished work
pauses rather than inventing completion. Preserve streamed answers in final output.
Move durable instruction writes outside the global control lock, reserve in-flight
acceptance against sealing, and roll back in-memory enqueue on persistence failure.
Reject ignored/credential paths for direct reads too. Unknown native tool names
receive bounded regeneration recovery without dispatch. Corrupt checkpoints still
remain intact and require a new chat; failed goals remain resumable by user request.
# Final review corrections (5579390225)

Tool-only malformed generations no longer append an empty assistant message to
the retry request. Repeated prose pauses now explain how to continue. Enter,
button submission, and the submit handler share one readiness predicate.
Recent approval-control protocol blocks get bounded priority during eviction;
only trusted goal checkpoints restore their internal preservation hint. Oversize
blocks still obey the context cap; server permission enforcement is independent.
Turn transcript replacement and turn accounting share the runtime lock. Direct
answer completion summaries are capped at the same 4000 characters as the tool.
The browser exhaustion test asserts the actionable recovery error; a separate
test covers successful regeneration and persistence.
