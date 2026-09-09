# Per-chat code workspaces

Stack on the fully integrated agentic harness (#353).

During parent integration, retain invocation-bound per-call approval for every
durable write/edit/terminal call, even in auto-approve mode. Reconcile the new
workspace argument with the parent session owner's bounded lifecycle: validate
mount sources before runtime calls, pin each tracked scope to its original root,
and keep ownership labels, deadlines, capacity limits, cleanup, and bounded I/O.
Preserve configured-root validation and safe public error messages. The owner
identifier is now ToolContext.workspace_id; coding_workspace_id is only a goal
label and must not select or rotate a chat directory.

When no explicit directory is configured, use
`<Geist data dir>/code/workspace-<owner ID>/chat-<chat ID>` for coding tools.
Allocate a chat row before tools execute to give new chats a stable identity.
Keep that directory across new runs, goals, and sandbox recreation. Existing
explicit GEIST_WORKSPACE_ROOT / GEIST_EXEC_WORKSPACE overrides remain unchanged.

File and terminal tools resolve the same directory. Docker mounts only that
chat's leaf at /workspace, with existing network/capability/user restrictions;
terminal execution requires fresh approval because storage is durable. Managed
directory symlinks are rejected. No schema migration or new dependencies.

The backend and Docker daemon must see the source at the same absolute path.
For containerized backends, operators must share the data directory accordingly;
missing daemon paths fail rather than creating an unrelated directory. Native
local execution remains unsandboxed, as before; directory separation is not an
OS security boundary for that explicitly selected backend.

Do not remove or migrate existing sandbox files automatically. Verify per-chat
and owner separation, restart persistence, explicit overrides, protected
approvals, pre-tool chat identity, and Docker bind arguments with focused tests.
# Final review dispositions (5579416196)

Chat allocation failure emits a distinct, sanitized database/workspace error
before model or tool execution. Cleanup and late custom-client promotion use
CodingWorkspace.scope rather than a mismatched string format. Missing Docker or
Podman hides file tools and yields the correct runtime remedy when called
directly. Managed-path validation uses model-safe WorkspaceOperationError.
Document Linux bind-file ownership and the absence of a chat-deletion endpoint.
Early identity allocation before permission loading is deliberate: failures
retain the chat identity and deny execution. The managed-workspace posture is
set during registry construction before tool registration; a constructor-only
refactor is optional and does not change that enforced ordering.
