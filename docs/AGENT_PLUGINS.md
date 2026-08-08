# Agent Plugins

Geist consumes plugins in the [Agent Plugins](https://agent-plugins.org) 1.0.0
format — the vendor-neutral packaging standard for agent extensions also used
by Claude Code, ChatGPT, Cursor, GitHub Copilot, and VS Code. Geist implements
the portable core of the standard: the `plugin.json` manifest, markdown skills,
and MCP server declarations. Host-specific extensions found in richer
manifests (hooks, subagents, LSP servers, themes, `userConfig`) are ignored,
so plugins written for those hosts still load — their portable components just
work and the rest is skipped.

## Installing a plugin

Copy (or clone) a plugin directory into the plugin root:

- `GEIST_PLUGIN_DIR` if set, otherwise
- `<data dir>/plugins` (for example `~/.local/share/geist/plugins` on Linux).

Each immediate subdirectory is one plugin:

```
plugins/
└── my-plugin/
    ├── .claude-plugin/
    │   └── plugin.json          # manifest (also accepted at the plugin root)
    ├── skills/
    │   └── code-review/
    │       └── SKILL.md         # YAML frontmatter + markdown instructions
    └── .mcp.json                # optional MCP server declarations (or mcp.json)
```

`GET /api/v1/plugins` lists what was discovered; `POST /api/v1/plugins/refresh`
re-scans the directory without a restart.

## Manifest

`name` is required and must be kebab-case (`^[a-z0-9][a-z0-9_-]{0,63}$`).
`version`, `description`, and `displayName` are read when present; `skills`
may add extra skill directories (paths must stay inside the plugin); and
`mcpServers` may declare servers inline or point at a JSON file inside the
plugin. All other fields are ignored per the standard.

## Skills

Every `skills/*/SKILL.md` with a `description` in its frontmatter becomes an
installed skill, namespaced `<plugin>:<skill>`. Skills use progressive
disclosure: the chat system prompt lists only names and descriptions, and the
model fetches a skill's full markdown body on demand through the built-in
`skills.load` tool. Skills with `disable-model-invocation: true` are parsed
but never advertised to the model.

## MCP servers

Plugin-declared MCP servers reuse the same client, discovery cache, and tool
registry machinery as servers configured through `/api/v1/mcp` (see
`CHAT_TOOL_REGISTRY.md`). `${CLAUDE_PLUGIN_ROOT}` (or `${GEIST_PLUGIN_ROOT}`)
in commands, args, env, URLs, and headers expands to the plugin's absolute
path. Tools mount as `mcp.<plugin>.<server>.<tool>` with the `external_write`
side-effect label.

## Trust model

Placing a plugin in the plugin directory is the install action and implies
trust of its *skills* — they are passive markdown read on demand. MCP servers
spawn processes or open network connections, so they are held to the same
posture as operator-configured servers: disabled by default. Enable a
plugin's servers by naming the plugin in `GEIST_ENABLED_PLUGINS`
(comma-separated), mirroring `GEIST_ENABLED_CHAT_TOOLS`. Skill bodies and MCP
tool descriptions/results are third-party content: treat them as untrusted
input, not instructions to the operator.
