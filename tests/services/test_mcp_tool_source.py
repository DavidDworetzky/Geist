"""Tests for mounting MCP server tools into the unified ToolRegistry."""

import datetime
import threading

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

    def call_tool(self, config, name, arguments, **kwargs):
        self.calls.append((config.name, name, arguments))
        return f"{name} says: {arguments.get('text', '')}"


def _context() -> ToolContext:
    return ToolContext(user_id=1, chat_id=None, run_id="run-test")


def _approved_context(call: ToolCall) -> ToolContext:
    return ToolContext(
        user_id=1,
        chat_id=None,
        run_id="run-test",
        approved_call_ids=frozenset({call.id}),
    )


def test_definitions_namespace_and_schema_passthrough():
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda user_id=None: [_server_model()])

    definitions = source.definitions()
    assert [definition.name for definition in definitions] == ["mcp.fake.echo"]
    definition = definitions[0]
    assert definition.arguments_schema == ECHO_TOOL["inputSchema"]
    assert definition.parameters_schema()["required"] == ["text"]
    assert definition.side_effect == "external_write"
    assert definition.requires_approval is True
    assert "[MCP: fake]" in definition.description


def test_registry_executes_mcp_tool_through_source():
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda user_id=None: [_server_model()])
    registry = ToolRegistry()
    registry.add_source(source)

    call = ToolCall.create("mcp.fake.echo", {"text": "hello"})
    unapproved = registry.execute(call, _context())

    assert unapproved.status == "awaiting_approval"
    assert unapproved.error == "approval_required"
    assert manager.calls == []

    result = registry.execute(call, _approved_context(call))

    assert result.status == "succeeded"
    assert result.content == "echo says: hello"
    assert manager.calls == [("fake", "echo", {"text": "hello"})]


def test_registry_rejects_missing_required_argument():
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda user_id=None: [_server_model()])
    registry = ToolRegistry()
    registry.add_source(source)

    call = ToolCall.create("mcp.fake.echo", {})
    result = registry.execute(call, _approved_context(call))

    assert result.status == "failed"
    assert result.error == "invalid_arguments"
    assert manager.calls == []


def test_registry_rejects_wrong_argument_type():
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda user_id=None: [_server_model()])
    registry = ToolRegistry()
    registry.add_source(source)

    call = ToolCall.create("mcp.fake.echo", {"text": 7})
    result = registry.execute(call, _approved_context(call))

    assert result.status == "failed"
    assert result.error == "invalid_arguments"


def test_registry_rejects_approval_after_mcp_configuration_changes():
    server = _server_model()
    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=lambda user_id=None: [server])
    registry = ToolRegistry()
    registry.add_source(source)
    call = ToolCall.create("mcp.fake.echo", {"text": "hello"})
    approved_definition = registry.get(call.name, _context())
    assert approved_definition is not None

    server.env["CHANGED"] = "after-approval"
    source.invalidate()
    result = registry.execute(
        call,
        _approved_context(call),
        expected_approval_fingerprint=approved_definition.approval_fingerprint(),
    )

    assert result.status == "failed"
    assert result.error == "approval_stale"
    assert manager.calls == []


def test_failing_server_is_skipped():
    manager = FakeManager(
        tools_by_server={"good": [ECHO_TOOL]},
        errors_by_server={"bad": "connection refused"},
    )
    source = McpToolSource(
        manager,
        servers_loader=lambda user_id=None: [
            _server_model(1, "bad"),
            _server_model(2, "good"),
        ],
    )

    definitions = source.definitions()
    assert [definition.name for definition in definitions] == ["mcp.good.echo"]


def test_definitions_cached_until_invalidated():
    loads = []

    def loader(user_id=None):
        loads.append(user_id)
        return [_server_model()]

    manager = FakeManager(tools_by_server={"fake": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=loader, cache_ttl_seconds=3600)

    source.definitions()
    source.definitions()
    assert len(loads) == 1

    source.invalidate()
    source.definitions()
    assert len(loads) == 2


def test_definitions_cache_and_loading_are_scoped_per_user():
    loaded_user_ids = []

    def loader(user_id=None):
        loaded_user_ids.append(user_id)
        return [_server_model(user_id or 0, f"user-{user_id}")]

    manager = FakeManager(tools_by_server={"user-1": [ECHO_TOOL], "user-2": [ECHO_TOOL]})
    source = McpToolSource(manager, servers_loader=loader, cache_ttl_seconds=3600)

    user_1 = ToolContext(user_id=1, chat_id=None, run_id="one")
    user_2 = ToolContext(user_id=2, chat_id=None, run_id="two")
    assert source.definitions(user_1)[0].name == "mcp.user-1.echo"
    assert source.definitions(user_2)[0].name == "mcp.user-2.echo"
    source.definitions(user_1)
    assert loaded_user_ids == [1, 2]


def test_invalidation_prevents_stale_discovery_from_repopulating_cache():
    first_started = threading.Event()
    release_first = threading.Event()

    class RacingManager(FakeManager):
        def __init__(self):
            super().__init__()
            self.discovery_count = 0

        def list_tools(self, config):
            self.discovery_count += 1
            if self.discovery_count == 1:
                first_started.set()
                assert release_first.wait(2)
                return [{**ECHO_TOOL, "name": "stale"}]
            return [{**ECHO_TOOL, "name": "fresh"}]

    manager = RacingManager()
    source = McpToolSource(manager, servers_loader=lambda user_id=None: [_server_model()])
    stale_result = []
    stale_thread = threading.Thread(
        target=lambda: stale_result.extend(source.definitions()),
        daemon=True,
    )
    stale_thread.start()
    assert first_started.wait(1)

    source.invalidate()
    fresh = source.definitions()
    release_first.set()
    stale_thread.join(timeout=2)

    assert [definition.name for definition in fresh] == ["mcp.fake.fresh"]
    assert [definition.name for definition in source.definitions()] == ["mcp.fake.fresh"]


def test_discovery_is_single_flight_per_user_and_generation():
    first_started = threading.Event()
    release = threading.Event()

    class BlockingManager(FakeManager):
        def __init__(self):
            super().__init__()
            self.discovery_count = 0

        def list_tools(self, config):
            self.discovery_count += 1
            first_started.set()
            assert release.wait(2)
            return [ECHO_TOOL]

    manager = BlockingManager()
    source = McpToolSource(manager, servers_loader=lambda user_id=None: [_server_model()])
    results: list[list] = []
    threads = [
        threading.Thread(target=lambda: results.append(source.definitions()), daemon=True)
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    assert first_started.wait(1)
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert manager.discovery_count == 1
    assert len(results) == 3


def test_config_from_model_maps_fields():
    config = config_from_model(_server_model())
    assert config.server_id == 1
    assert config.name == "fake"
    assert config.transport == "stdio"
    assert config.command == "fake-command"
    assert config.timeout_seconds == 10.0
