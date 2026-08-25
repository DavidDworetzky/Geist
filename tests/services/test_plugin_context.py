"""Tests for skill prompt context and plugin tool/MCP mounting."""

import json

from agents.models.tool_calling import ToolCall, ToolContext
from app.services.plugin_context import build_plugin_skills_context, install_plugin_support
from app.services.plugin_loader import PluginRegistry
from app.services.tool_registry import ToolRegistry


def _write_plugin(root, name="demo-plugin", mcp_servers=None):
    plugin_root = root / name
    manifest_dir = plugin_root / ".claude-plugin"
    manifest_dir.mkdir(parents=True)
    manifest = {"name": name}
    if mcp_servers is not None:
        manifest["mcpServers"] = mcp_servers
    (manifest_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    return plugin_root


def _write_skill(plugin_root, name="demo-skill", description="Demo description", extra=""):
    skill_dir = plugin_root / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\nSkill body text.\n",
        encoding="utf-8",
    )


def _context() -> ToolContext:
    return ToolContext(workspace_id=1, chat_id=None, run_id="run-test")


class TestSkillsContext:
    def test_empty_without_plugins(self, tmp_path):
        assert build_plugin_skills_context(PluginRegistry(tmp_path)) == ""

    def test_lists_qualified_names_and_descriptions(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root)
        context = build_plugin_skills_context(PluginRegistry(tmp_path))
        assert "demo-plugin:demo-skill" in context
        assert "Demo description" in context
        assert "skills.load" in context

    def test_model_hidden_skills_are_excluded(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root, extra="disable-model-invocation: true\n")
        assert build_plugin_skills_context(PluginRegistry(tmp_path)) == ""


class TestSkillLoadTool:
    def test_loads_skill_body(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root)
        tool_registry = ToolRegistry()
        install_plugin_support(tool_registry, PluginRegistry(tmp_path))

        result = tool_registry.execute(
            ToolCall(
                id="call-1", name="skills.load", arguments={"skill": "demo-plugin:demo-skill"}
            ),
            _context(),
        )
        assert result.succeeded
        assert "Skill body text." in result.content

    def test_unknown_skill_fails(self, tmp_path):
        plugin_root = _write_plugin(tmp_path)
        _write_skill(plugin_root)
        tool_registry = ToolRegistry()
        install_plugin_support(tool_registry, PluginRegistry(tmp_path))

        result = tool_registry.execute(
            ToolCall(id="call-2", name="skills.load", arguments={"skill": "demo-plugin:missing"}),
            _context(),
        )
        assert result.status == "failed"

    def test_tool_registered_even_without_plugins(self, tmp_path):
        tool_registry = ToolRegistry()
        install_plugin_support(tool_registry, PluginRegistry(tmp_path))
        assert tool_registry.get("skills.load") is not None


class TestPluginMcpMounting:
    def test_plugin_mcp_source_added_with_distinct_name(self, tmp_path):
        tool_registry = ToolRegistry()
        install_plugin_support(tool_registry, PluginRegistry(tmp_path))
        source_names = [source.name for source in tool_registry._sources]
        assert "plugin-mcp" in source_names

    def test_coexists_with_db_backed_mcp_source(self, tmp_path):
        from app.services.mcp_client import McpClientManager
        from app.services.mcp_tool_source import McpToolSource

        tool_registry = ToolRegistry()
        tool_registry.add_source(McpToolSource(McpClientManager(), servers_loader=list))
        install_plugin_support(tool_registry, PluginRegistry(tmp_path))
        assert len(tool_registry._sources) == 2

    def test_enabled_plugin_servers_mount_tools(self, tmp_path, monkeypatch):
        from app.services.mcp_tool_source import McpToolSource

        _write_plugin(
            tmp_path,
            "demo-plugin",
            mcp_servers={"srv": {"command": "fake-server"}},
        )
        monkeypatch.setenv("GEIST_ENABLED_PLUGINS", "demo-plugin")
        plugin_registry = PluginRegistry(tmp_path)

        class FakeManager:
            def list_tools(self, config):
                assert config.name == "demo-plugin.srv"
                assert config.server_id < 0
                return [
                    {
                        "name": "echo",
                        "description": "Echo text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    }
                ]

            def call_tool(self, config, name, arguments, **_kwargs):
                return f"{name}:{arguments['text']}"

        source = McpToolSource(
            FakeManager(), servers_loader=plugin_registry.enabled_mcp_server_models
        )
        source.name = "plugin-mcp"
        tool_registry = ToolRegistry()
        tool_registry.add_source(source)

        definitions = {definition.name for definition in tool_registry.catalog()}
        assert "mcp.demo-plugin.srv.echo" in definitions

        call = ToolCall(
            id="call-3",
            name="mcp.demo-plugin.srv.echo",
            arguments={"text": "hello"},
        )
        result = tool_registry.execute(
            call,
            ToolContext(
                workspace_id=1,
                chat_id=None,
                run_id="run-test",
                approved_call_ids=frozenset({call.id}),
            ),
        )
        assert result.succeeded
        assert result.content == "echo:hello"
