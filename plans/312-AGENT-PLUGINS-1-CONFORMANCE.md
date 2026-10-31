# Agent Plugins 1.0.0 Conformance

## Goal

Make PR #312 a conformant Agent Plugins 1.0.0 client for the portable core it supports: Agent Skills plus stdio and Streamable HTTP MCP servers. Preserve Geist's operator authorization, workspace scoping, progressive disclosure, and shared MCP approval/runtime protections.

## Normative contract

- Load the required root `plugin.json` and select the locally supported rules from the canonical 1.0.0 `$schema` identifier.
- Validate the closed manifest schema, including plugin names, metadata types, and extension-object boundaries. Report and ignore unknown top-level fields; reject all other manifest violations before component discovery.
- Discover only root `skills/` and root `mcp.json` as portable components. Missing locations are valid; invalid component locations fail independently.
- Validate each `SKILL.md` against the Agent Skills name, parent-directory, description, and frontmatter requirements while isolating invalid skills.
- Validate the closed `mcp.json` document and each server entry independently. Support `stdio` and `streamable-http`; skip optional legacy `sse` without affecting other components.
- Enforce executable-token, plugin/data path containment, remote URL, literal-header, and matching-schema-version rules.
- Provide `PLUGIN_ROOT` and a dedicated persistent `PLUGIN_DATA` directory to plugin subprocesses; expand only those placeholders in args, environment values, and cwd; use the plugin root as the default cwd.

## Implementation

1. Replace the legacy/Claude-oriented loader rules with explicit Agent Plugins 1.0.0 validation and fixed component discovery.
2. Extend plugin and MCP runtime models with the cwd and reserved plugin environment context needed by stdio transports without changing database persistence for operator-configured MCP servers.
3. Keep unsupported client extensions inert and keep legacy layouts out of the portable conformance path.
4. Update documentation and API fixtures to show canonical `plugin.json`, `mcp.json`, `$schema`, skill frontmatter, and declared transports.

## Verification

- Add positive and negative conformance tests derived from the official plugin and MCP schemas and Agent Skills requirements.
- Run plugin loader/context/API tests, MCP client/source tests, operator security tests, formatting, Ruff, and mypy.
- Run the broader stacked backend suite and the applicable Geist pre-push runtime checks.
- Push the refreshed head, monitor CI, and post an exact conformance matrix on PR #312.
