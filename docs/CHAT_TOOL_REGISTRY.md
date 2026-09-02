# Native chat tool registry

Interactive chats use a small Python contract in `agents/models/tool_calling.py`.
`ChatOrchestrator` sends only reviewed JSON schemas to a capable model backend,
executes exact registry handlers, appends normalized tool results to the model
transcript, and persists the turn once. It does not route through the legacy
adapter reflection mechanism.

`ToolRegistry` is the single registry for every tool exposed to chat. Besides
the statically registered defaults below, it accepts pluggable *tool sources*
(`ToolSource` in `app/services/tool_registry.py`) whose definitions are merged
into the catalog on every turn: MCP servers (`app/services/mcp_tool_source.py`)
and bridged adapter actions (`app/services/adapter_tool_source.py`). Statically
registered names win collisions, and a failing source contributes no tools
instead of breaking chat. Source-provided tools carry raw JSON schemas
(`ToolDefinition.arguments_schema`); the registry checks required keys and
primitive types before dispatch and leaves full constraint enforcement to the
tool's own backend.

The default registry is intentionally explicit:

| Tool | Backing implementation | Default | Policy |
| --- | --- | --- | --- |
| `web.search` | `SearchAdapter.search` | yes | Bounded public search results; arbitrary URL fetch is not exposed. |
| `documents.search` | `DocumentSearchService.search` | yes | Read-only and scoped to the current workspace's uploaded files. |
| `image.generate` | `ImageGenerationAdapter.generate_image` | when an OpenAI image key is configured | Cost-bearing network write; intended only for explicit image requests. |
| `workspace.list_markdown` | `MarkdownFileAdapter.get_files` | yes | Read-only paths contained under Geist's private workspace directory. |
| `workspace.read_markdown` | `MarkdownFileAdapter.read_file` | yes | Read-only paths contained under Geist's private workspace directory. |

The read-only Markdown list/read tools default to
`GEIST_DATA_DIR/workspace` (`~/Library/Application Support/Geist/workspace` on
macOS, `~/.local/share/geist/workspace` on Linux, and `%LOCALAPPDATA%/Geist/workspace`
on Windows). Operators can select a different directory with
`GEIST_MARKDOWN_ROOT`. Reads accept only `.md`/`.markdown` paths contained under
that root.

Adapters not mapped into chat tools:

- `SearchAdapter.get`: excluded because open-ended URL fetching creates an SSRF boundary.
- `LogAdapter`: internal diagnostics, not user chat data.
- `MMSAdapter`: internal voice preprocessing and expensive model initialization.
- `WhisperAdapter`: incomplete legacy stub; it must be replaced before registration.
- adapter constructors, private helpers, and `enumerate_actions`: never tools.

Provider behavior:

- `OnlineAgent` normalizes OpenAI-compatible streamed function calls and
  Anthropic `tool_use` blocks into the same contract. Internal dotted names
  are reversibly encoded to provider-safe function names and mapped back before
  registry dispatch.
- Native requests retry only transient failures that occur before any event is
  emitted. Configured backups use their own URL, model, and credentials; when a
  turn includes tools, unknown compatible endpoints are skipped unless native
  tool capability is explicitly declared.
- Custom online endpoints do not receive tool schemas by default. They can opt
  into the capability explicitly; known OpenAI, Anthropic, Groq, and xAI
  endpoints are recognized automatically.
- `LocalAgent` currently advertises no tools. Local runners fail closed until a
  runner-specific native tool template and parser is implemented.
- Existing non-streaming completion clients remain text-only by default. The
  chat frontend explicitly sends `enable_tools: true` to the streaming routes;
  API clients can make the same opt-in.

The browser contract is SSE: `run_started`, repeated `delta`, upsert-style
`tool_call` states, `artifact`, `final`, `error`, and `done`. Active runs can be
cooperatively cancelled with `POST /agent/runs/{run_id}/cancel`; the reviewed
catalog is available from `GET /agent/tools`. The UI confirms cancellation only
after the server accepts it, treats premature stream EOF as failure, and keeps
run output associated with the chat in which the run started.

Every terminal run is persisted with `completed`, `failed`, or `cancelled`
status so completed tool activity is auditable even if a later model request
fails. Recent-history and aggregate tool-result budgets bound provider context.
Tool execution uses a bounded worker pool. Cancellation is cooperative: MCP
stdio requests observe the run cancellation event, while HTTP operations are
bounded by the same end-to-end server deadline and have their connection closed
when invalidated. Side-effect mappings additionally require an interactive,
call-specific approval before dispatch.

## Security and identity model

Geist deliberately separates three identities that older code conflated:

- The **workspace identity** is the durable local owner of chats, memories,
  files, settings, workflows, jobs, and MCP configuration. It is the neutral
  `geist_user` row with `workspace_key="default"`; its numeric primary key is
  preserved when older databases are migrated. The physical table and several
  legacy foreign keys retain the `user_id` column name for migration compatibility,
  but application-facing ownership contracts use `workspace_id`. Display name and email are
  profile metadata, not credentials, and the legacy password column is not an
  authentication mechanism.
- The **operator principal** is the process or person authorized to control the
  local workspace. It is immutable and request-scoped, carries `workspace_id`,
  an authentication method, loopback status, and explicit capabilities. It can
  also carry controller-node, target-node, audience, expiry, and credential IDs
  for a future multi-node deployment. There is no built-in admin user account
  and no email-based login.
- **Pitchblend Cloud identity** remains Pitchblend's OIDC login for account,
  licensing, and sync. Its access and refresh tokens are not accepted by Geist
  and never become local tool credentials.

An unwrapped standalone Geist process accepts an operator only from a loopback
client. The supported Docker Compose and native `make run` flows generate a
local operator-token file. The backend and server-side development proxy read
that file, and the proxy attaches the `GeistOperator` scheme without exposing
the credential to the browser bundle. Compose publishes its UI, API, and debug
ports on loopback only. A token-file principal is therefore local even when the
development proxy reaches the backend over a private container address.

A Pitchblend-managed launch instead generates a high-entropy secret for that
process, passes it to the Geist child as `GEIST_OPERATOR_TOKEN`, and attaches
the same authorization scheme inside a dedicated Electron webview session.
The secret is not exposed to page JavaScript. When either supported credential
source is configured, missing, duplicate, malformed, and incorrect credentials
fail closed; conflicting environment and file credentials also fail closed.
Pitchblend login and logout do not rotate workspace ownership or require a
second Geist login.

MCP configuration, connection tests, tool catalog inspection, approval, and
run cancellation require operator capabilities. Stdio server configuration is
accepted only from a local operator: a direct loopback request or the generated
local token-file principal. A future remote wrapper-token deployment remains
unable to configure process-spawning transports. A remote deployment must
provide its own strong operator credential and transport boundary; it must not
treat a Host or Origin header as authentication.

Approval is authorization for one exact operation, not a reusable boolean.
The pending record binds the operator's `workspace_id`, run ID, call ID, tool name,
canonical argument hash, and tool-definition fingerprint. The definition
fingerprint includes the MCP configuration revision. If arguments or server
configuration change between review and dispatch, execution fails with a stale
approval and the operator must review again. Non-streaming callers are always
unattended and deny approval-requiring tools immediately instead of waiting for
the interactive timeout.

The authentication middleware protects HTTP and WebSocket requests globally;
only health, readiness, and generated API-documentation endpoints are exempt.
For a future Tailscale topology, each workstation remains a separate Geist node
with its own local workspace and node-scoped credential. Pitchblend is the
controller: it authenticates the human with OIDC, chooses a target node, and
presents a short-lived operator credential whose audience and `target_node_id`
match that Geist instance. Tailscale supplies encrypted reachability and device
policy, but is defense in depth rather than the application authentication
decision. Node credentials should not be shared or interpreted as a global
multi-user database identity.

## MCP servers

Operators can mount external tools from MCP (Model Context Protocol) servers,
configured in Settings → MCP Servers or via `/api/v1/mcp/servers`. Two
transports are supported: `stdio` (a locally spawned process) and `http`
(streamable HTTP). Configured servers are persisted in the `mcp_server` table
and **start disabled**; only explicitly enabled servers contribute tools, and
a test-connection endpoint (`POST /api/v1/mcp/servers/{id}/test`) lists a
server's tools without enabling it.

Mounted tools are named `mcp.<server>.<tool>`, labelled with the
`external_write` side effect, and executed through the same bounded worker
pool, timeouts, and result budgets as every other chat tool. The client in
`app/services/mcp_client.py` is a deliberately minimal synchronous JSON-RPC
implementation (initialize, paginated `tools/list`, `tools/call`); server
initiated requests such as sampling are declined. Tool descriptions and
results from MCP servers are untrusted third-party content — enable only
servers you trust, since their tool output re-enters the model context.

The transport boundary also:

- inherits only a small process-launch environment allowlist; secrets are
  passed to stdio servers only when explicitly configured for that server;
- rejects HTTP redirects and credential-bearing URLs so an MCP endpoint cannot
  redirect an authorized request into a different SSRF target;
- applies one deadline across initialization and paginated discovery rather
  than resetting the timeout on each page;
- caps response bytes, SSE events, stdio line length, page count, discovered
  tool count, and returned tool-result text;
- performs server discovery off the FastAPI event loop, in bounded parallel
  workers, with generation-aware single-flight caches so invalidation cannot
  publish a stale tool definition or connection.

## Bridged adapter actions

`AdapterToolSource` converts adapter reflection schemas
(`adapters/tool_schema.py`) into ordinary registry definitions named
`adapter.<Adapter>.<action>`, replacing the need for a second dispatch
mechanism. Bridged actions are disabled by default and must be opted into by
name through `GEIST_ENABLED_CHAT_TOOLS`; the default registry bridges only the
read-only `JobStatusAdapter`. Migrating the legacy agent tick loop onto the
unified registry is follow-up work.

## Doom-loop interrupt

`ChatOrchestrator` interrupts a run when the model issues the same tool call
(identical name and arguments) three times consecutively. The repeated call is
not executed; the run fails with a `Doom loop detected` error so a stuck model
cannot burn its round and tool budgets on identical requests.
