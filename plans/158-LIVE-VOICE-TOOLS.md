# Live voice tools

This change was stacked on `codex/live-voice-moshi` (PR #371), now merged into
`main`. Keep tool integration in its own PR, separate from the audio transport
and local Moshi capability.

GPT-Live client delegation will use the existing authenticated chat streaming
endpoint, selected text model, tool registry, argument validation, workspace
permissions, approval registry, cancellation, and persisted results. Do not add
a second tool dispatcher or infer approval from spoken transcripts. Moshi's
audio-only protocol is unchanged.

Add a default-off `tools_enabled` field to live session creation. The browser
enables it when its tool handler is present and explains the selected text-model
dependency in voice settings. Session instructions describe delegation and
require verified results before claiming actions succeeded.

Collect bounded input/output transcripts in the browser. On client delegation,
deduplicate the delegation ID, require new user transcript, and run at most one
task at a time. Use a separate chat for each voice call with memory disabled;
reuse it for subsequent tasks so completed actions remain in context. Return
bounded factual results through `session.commentary.append` with the original
delegation ID. Keep full output and tool approvals visible in the UI. Closing
the call aborts the stream and cancels pending approvals through existing server
disconnect handling. Never retry a failed or interrupted action automatically.

Verify session request compatibility, transcript/context handling, duplicates,
overlapping tasks, failure/closure, result size, real registry dispatch and
approval behavior, frontend type/lint checks, and the unchanged voice controls.
Run Docker checks if available and document unavailable paid voice validation.

Protocol reference: https://developers.openai.com/api/docs/guides/live-delegation
