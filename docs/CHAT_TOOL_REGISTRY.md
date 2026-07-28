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
| `documents.search` | `DocumentSearchService.search` | yes | Read-only and scoped to the current user's uploaded files. |
| `image.generate` | `ImageGenerationAdapter.generate_image` | when an OpenAI image key is configured | Cost-bearing network write; intended only for explicit image requests. |
| `workspace.list_markdown` | `MarkdownFileAdapter.get_files` | no | Paths are contained under `GEIST_MARKDOWN_ROOT`. |
| `workspace.read_markdown` | `MarkdownFileAdapter.read_file` | no | Paths are contained under `GEIST_MARKDOWN_ROOT`. |
| `workspace.write_markdown` | `MarkdownFileAdapter.write_file` | unavailable | Catalogued mapping; blocked until approval/resume and idempotency exist. |
| `communication.email.send` | `SendGridAdapter.send_email` | unavailable | Catalogued mapping; blocked until approval/resume and idempotency exist. |
| `communication.sms.send` | `SMSAdapter.send_text` | unavailable | Catalogued mapping; blocked until approval/resume and idempotency exist. |

The optional read-only Markdown list/read tools can be selected by the operator
with a comma-separated `GEIST_ENABLED_CHAT_TOOLS` value. Reads and writes accept
only `.md`/`.markdown` paths contained under the configured root. Side-effect
mappings remain unavailable even when named in that setting.

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
Tool execution uses a bounded worker pool; Python cannot forcibly stop a handler
already running after a timeout, which is another reason side-effect mappings
remain unavailable.

A future authenticated approve/reject-and-resume endpoint plus durable
idempotency is required before filesystem writes, email, or SMS can become
operational chat tools.

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
