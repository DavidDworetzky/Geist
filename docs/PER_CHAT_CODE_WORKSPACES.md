# Default code directories

Without a configured code directory, each chat stores its files under:

```
<Geist data directory>/code/workspace-<owner ID>/chat-<chat ID>/
```

The data directory follows `GEIST_DATA_DIR` or the platform's Geist application
data location. A new chat gets its database identity before its first tool call.
Files survive later messages, new goals, application restarts, and removal of
idle sandbox containers. They are not automatically deleted with those containers.
Separate chats and owners get separate directories. An explicit
`GEIST_WORKSPACE_ROOT` or `GEIST_EXEC_WORKSPACE` continues to override the default;
chats using the same explicit directory intentionally share its files.

The coding tools (`workspace.list_files`, `read_file`, `search`, `write_file`,
`edit_file`) and `terminal.run` use the same directory. This does not relocate
the separate Markdown/document library tools.

Docker/Podman binds only the chat's directory at `/workspace`, retaining its
read-only container root, non-root user, dropped capabilities, and default
disabled network. File mutations and terminal commands require fresh approval
even with auto-approve or always-allow selected, because they can change durable
host files. Consequently, unattended scheduled routines cannot run terminal
commands or write/edit code files. They can still use eligible read-only tools.
There is no routine-level standing consent for durable writes; run such tasks
interactively when approval is needed. Choosing the `local` execution backend still runs unsandboxed on the
host: a working directory is not a security boundary for local shell commands.

The backend and container daemon must see the directory at the same absolute
path. With a containerized backend or remote daemon, arrange that shared path
explicitly. Missing daemon-side paths fail closed; Geist does not auto-create a
different daemon-host directory. Native Geist with a local Docker daemon uses
the normal host bind-mount workflow.

Existing files in older ephemeral sandboxes are not moved or removed by this
change. Export them before removing those old containers. Custom tool clients
without a chat identity receive a run-scoped fallback directory instead.

Use a private, trusted `GEIST_DATA_DIR`, not an untrusted shared location.
Geist enforces owner-only permissions on managed `code` and `workspace-<id>`
ancestors on each access; operator changes to those modes are intentionally
overridden. The sandbox leaf permits UID 65534 writes, so its private ancestors
are part of the host protection. Explicit directory overrides are not chmodded.

Managed directories have no automatic retention sweep. Scheduled runs allocate
new chat identities; code directories are created lazily when a coding tool uses
them. Files accumulate until an operator archives or removes them. Early startup
failures can leave empty chat rows; they contain no generated files unless a
coding tool ran. No automatic cleanup is added here because it could delete work.

On native Linux bind mounts, files created by the Docker sandbox may be owned by
UID 65534 with mode 0644. Editing those files directly, or changing an existing
chat from Docker to local execution, may require an operator ownership repair.
Docker Desktop ownership behavior may differ. Geist never runs automatic chown.
Chat container cleanup uses the same namespaced key as execution. There is no
chat-deletion endpoint yet; idle expiry and process shutdown reclaim containers,
not generated files. Custom clients that persist a chat late promote their run
scope through the same key function.
