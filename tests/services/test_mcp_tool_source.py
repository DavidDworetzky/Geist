"""Tests for mounting MCP server tools into the unified ToolRegistry."""

import datetime

from agents.models.tool_calling import ToolCall, ToolContext
from app.models.database.mcp_server import McpServerModel
from app.services.mcp_client import McpError
from app.services.mcp_tool_source import McpToolSource, config_from_model
from app.services.tool_registry import ToolRegistry


ECHO_TOOL = {
    "name": "echo",
    "description": "Echo text back",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


def _server_model(server_id: int = 1, name: str = "fake") -> McpServerModel:
    now = datetime.datetime(2026, 7, 28)
    return McpServerModel(
        mcp_server_id=server_id,
        user_id=1,
        name=name,
        transport="stdio",
        command="fake-command",
        args=[],
        env={},
        url=None,
        headers={},
        enabled=True,
        timeout_seconds=10.0,
        create_date=now,
        update_date=now,
    )


class FakeManager:
    def __init__(self, tools_by_server=None, errors_by_server=None):
        self.tools_by_server = tools_by_server or {}
        self.errors_by_server = errors_by_server or {}
        self.calls = []

    def list_tools(self, config):
        if config.name in self.errors_by_server:
            raise McpError(self.errors_by_server[config.name])
        return self.tools_by_server.get(config.name, [])

    def call_tool(self, config, name, arguments):
        self.calls.append((config.name, name, arguments))
        return f"{name} says: {arguments.get('text', '')}"


def _context() -> ToolContext:
    return ToolContext(user_id=1, chat_id=None, run_id="run-test")


def test_definitions_namespace_and_schema_passthrough():
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda: [_server_model()])

    definitions = source.definitions()
    assert [definition.name for definition in definitions] == ["mcp.fake.echo"]
    definition = definitions[0]
    assert definition.arguments_schema == ECHO_TOOL["inputSchema"]
    assert definition.parameters_schema()["required"] == ["text"]
    assert definition.side_effect == "external_write"
    assert "[MCP: fake]" in definition.description


def test_registry_executes_mcp_tool_through_source():
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda: [_server_model()])
    registry = ToolRegistry()
    registry.add_source(source)

    call = ToolCall.create("mcp.fake.echo", {"text": "hello"})
    result = registry.execute(call, _context())

    assert result.status == "succeeded"
    assert result.content == "echo says: hello"
    assert manager.calls == [("fake", "echo", {"text": "hello"})]


def test_registry_rejects_missing_required_argument():
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda: [_server_model()])
    registry = ToolRegistry()
    registry.add_source(source)

    result = registry.execute(ToolCall.create("mcp.fake.echo", {}), _context())

    assert result.status == "failed"
    assert result.error == "invalid_arguments"
    assert manager.calls == []


def test_registry_rejects_wrong_argument_type():
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda: [_server_model()])
    registry = ToolRegistry()
    registry.add_source(source)

    result = registry.execute(ToolCall.create("mcp.fake.echo", {"text": 7}), _context())

    assert result.status == "failed"
    assert result.error == "invalid_arguments"


def test_failing_server_is_skipped():
    manager = FakeManager(
        tools_by_server={"good": [ECHO_TOOL]},
        errors_by_server={"bad": "connection refused"},
    )
    source = McpToolSource(
        manager,
        servers_loader=lambda: [_server_model(1, "bad"), _server_model(2, "good")],
    )

    definitions = source.definitions()
    assert [definition.name for definition in definitions] == ["mcp.good.echo"]


def test_definitions_cached_until_invalidated():
    loads = []

    def loader():
        loads.append(1)
        return [_server_model()]

    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=loader, cache_ttl_seconds=3600)

    source.definitions()
    source.definitions()
    assert len(loads) == 1

    source.invalidate()
    source.definitions()
    assert len(loads) == 2


def test_config_from_model_maps_fields():
    config = config_from_model(_server_model())
    assert config.server_id == 1
    assert config.name == "fake"
    assert config.transport == "stdio"
    assert config.command == "fake-command"
    assert config.timeout_seconds == 10.0
