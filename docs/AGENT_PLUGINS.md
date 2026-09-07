# Agent Plugins

Geist implements the portable core of the [Agent Plugins 1.0.0](https://agent-plugins.org/specification) standard: root `plugin.json`, Agent Skills under `skills/`, and stdio or Streamable HTTP servers in root `mcp.json`. Unsupported legacy SSE entries and client-extension namespaces are ignored without disabling independent portable components.

## Installing a plugin

Copy or clone a plugin directory into `GEIST_PLUGIN_DIR`, or into `<data dir>/plugins` when that variable is unset. Each immediate child directory is one plugin:

```text
plugins/
└── my-plugin/
    ├── plugin.json
    ├── skills/
    │   └── code-review/
    │       ├── SKILL.md
    │       ├── scripts/
    │       ├── references/
    │       └── assets/
    └── mcp.json
```

`GET /api/v1/plugins` lists discovered portable components. `POST /api/v1/plugins/refresh` rescans without restarting Geist. Both routes require the operator principal's `tools.manage` capability. Installation and execution belong to the operator's local workspace.

## Manifest

Every plugin requires root `plugin.json` with the canonical schema identifier and a valid name:

```json
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "Review and research workflows"
}
```

The closed portable manifest also permits `author`, `homepage`, `repository`, `license`, `keywords`, and `extensions`. Unknown top-level fields are reported and ignored; other schema violations reject the plugin. Geist does not treat `.claude-plugin/plugin.json` or manifest-level `skills`/`mcpServers` fields as portable Agent Plugins configuration.

## Skills

Every immediate `skills/*/SKILL.md` follows the Agent Skills specification. Its YAML frontmatter requires `name` and `description`; the name must match its parent directory. Optional standard fields are `license`, `compatibility`, `metadata`, and `allowed-tools`.

```markdown
---
name: code-review
description: Review changed code for correctness and security when preparing a PR.
---

# Code review

Inspect the diff and report findings by severity.
```

Geist namespaces the skill as `<plugin>:<skill>`. Chat uses progressive disclosure: the system prompt lists names and descriptions, and the model retrieves the full Markdown body through `skills.load` only when relevant. Invalid skills are skipped without disabling valid siblings or MCP servers.

## MCP servers

Root `mcp.json` requires the matching 1.0.0 schema and a closed `mcpServers` object:

```json
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
  "mcpServers": {
    "local-tools": {
      "type": "stdio",
      "command": "./bin/server",
      "args": ["--data", "${PLUGIN_DATA}"],
      "env": {"CONFIG": "${PLUGIN_ROOT}/config.json"},
      "cwd": "${PLUGIN_ROOT}"
    },
    "remote-tools": {
      "type": "streamable-http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

Geist supports `stdio` and `streamable-http`; optional legacy `sse` entries are skipped. Stdio processes receive client-owned `PLUGIN_ROOT` and persistent `PLUGIN_DATA` environment variables. Those placeholders expand only in args, environment values, and cwd. Bare commands use platform executable lookup; `./` commands and explicit working directories are contained within the plugin or data root. Non-loopback remote endpoints require HTTPS, and configured headers remain literal.

Plugin MCP tools reuse Geist's shared client, discovery cache, approvals, workspace scoping, and execution limits. Tools mount as `mcp.<plugin>.<server>.<tool>` with the `external_write` side-effect label.

## Trust model

Placing a plugin in the plugin directory is the installation decision for its passive Markdown skills. MCP servers can spawn processes or open network connections and remain disabled until the plugin name appears in `GEIST_ENABLED_PLUGINS`.

Discovery enforces package boundaries, rejects escaping symlinks and configured paths, bounds files and component counts, isolates invalid skills and MCP entries, and never passes ambient Geist secrets to plugin subprocesses. Plugin content and MCP results remain untrusted model input.
