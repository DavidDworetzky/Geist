"""Tests for the minimal synchronous MCP client (stdio transport + parsing)."""

import sys
import threading

import pytest

from app.services.mcp_client import (
    McpClientManager,
    McpConnection,
    McpError,
    McpServerConfig,
    _filtered_child_environment,
    _HttpTransport,
    _unwrap_response,
)


# A minimal MCP server speaking newline-delimited JSON-RPC over stdio.
FAKE_SERVER_SCRIPT = """
import json, sys

def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    method = message.get("method")
    message_id = message.get("id")
    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-server", "version": "1.0"},
            },
        })
    elif method == "tools/list":
        send({
            "jsonrpc": "2.0",
            "id": message_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text back",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            },
        })
    elif method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") == "echo":
            text = (params.get("arguments") or {}).get("text")
            send({
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {"content": [{"type": "text", "text": "echo: " + str(text)}]},
            })
        elif params.get("name") == "slow":
            pass  # never answer; used to exercise timeouts
        else:
            send({
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "unknown tool"}],
                },
            })
    elif message_id is not None:
        send({
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {"code": -32601, "message": "Method not found"},
        })
"""


def _stdio_config(timeout_seconds: float = 10.0) -> McpServerConfig:
    return McpServerConfig(
        server_id=1,
        name="fake",
        transport="stdio",
        command=sys.executable,
        args=("-c", FAKE_SERVER_SCRIPT),
        timeout_seconds=timeout_seconds,
    )


def test_stdio_initialize_list_and_call():
    connection = McpConnection(_stdio_config())
    try:
        assert connection.server_info.get("name") == "fake-server"
        tools = connection.list_tools()
        assert [tool["name"] for tool in tools] == ["echo"]
        assert tools[0]["inputSchema"]["required"] == ["text"]
        assert connection.call_tool("echo", {"text": "hello"}) == "echo: hello"
    finally:
        connection.close()


def test_stdio_tool_error_raises():
    connection = McpConnection(_stdio_config())
    try:
        with pytest.raises(McpError, match="unknown tool"):
            connection.call_tool("missing", {})
    finally:
        connection.close()


def test_stdio_request_timeout():
    connection = McpConnection(_stdio_config(timeout_seconds=1.0))
    try:
        with pytest.raises(McpError, match="timed out"):
            connection.call_tool("slow", {})
    finally:
        connection.close()


def test_stdio_request_honors_cancellation():
    connection = McpConnection(_stdio_config(timeout_seconds=5.0))
    cancellation = threading.Event()
    cancellation.set()
    try:
        with pytest.raises(McpError, match="cancelled"):
            connection.call_tool("slow", {}, cancellation=cancellation)
    finally:
        connection.close()


def test_stdio_child_environment_does_not_inherit_application_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("PATH", "/safe/bin")

    environment = _filtered_child_environment({"EXPLICIT_TOKEN": "configured"})

    assert environment["PATH"] == "/safe/bin"
    assert environment["EXPLICIT_TOKEN"] == "configured"
    assert "OPENAI_API_KEY" not in environment


def test_missing_command_fails_fast():
    config = McpServerConfig(server_id=2, name="broken", transport="stdio", command=None)
    with pytest.raises(McpError, match="no command"):
        McpConnection(config)


def test_unknown_transport_rejected():
    config = McpServerConfig(server_id=3, name="odd", transport="carrier-pigeon")
    with pytest.raises(McpError, match="Unknown MCP transport"):
        McpConnection(config)


def test_manager_reuses_and_invalidates_connections():
    manager = McpClientManager()
    config = _stdio_config()
    try:
        first = manager._connection(config)
        assert manager._connection(config) is first

        manager.invalidate(config.server_id)
        second = manager._connection(config)
        assert second is not first
        assert second.call_tool("echo", {"text": "again"}) == "echo: again"
    finally:
        manager.shutdown()


def test_manager_reconnects_when_fingerprint_changes():
    manager = McpClientManager()
    config = _stdio_config()
    try:
        first = manager._connection(config)
        changed = McpServerConfig(
            server_id=config.server_id,
            name=config.name,
            transport=config.transport,
            command=config.command,
            args=config.args,
            env={"EXTRA": "1"},
            timeout_seconds=config.timeout_seconds,
        )
        second = manager._connection(changed)
        assert second is not first
    finally:
        manager.shutdown()


def test_manager_invalidation_discards_connection_created_by_stale_flight(monkeypatch):
    import app.services.mcp_client as mcp_client

    first_started = threading.Event()
    release_first = threading.Event()
    created = []

    class FakeConnection:
        def __init__(self, config, *, deadline=None):
            self.config = config
            self.closed = False
            created.append(self)
            if len(created) == 1:
                first_started.set()
                assert release_first.wait(2)

        def close(self):
            self.closed = True

    monkeypatch.setattr(mcp_client, "McpConnection", FakeConnection)
    manager = McpClientManager()
    result = []
    worker = threading.Thread(
        target=lambda: result.append(manager._connection(_stdio_config())),
        daemon=True,
    )
    worker.start()
    assert first_started.wait(1)

    manager.invalidate(1)
    release_first.set()
    worker.join(timeout=2)

    assert len(created) == 2
    assert created[0].closed is True
    assert result == [created[1]]
    manager.shutdown()


def test_unwrap_response_surfaces_server_errors():
    with pytest.raises(McpError, match="boom .code -32000."):
        _unwrap_response(
            {"jsonrpc": "2.0", "id": "1", "error": {"code": -32000, "message": "boom"}},
            "tools/list",
        )
    assert _unwrap_response({"jsonrpc": "2.0", "id": "1", "result": {"ok": True}}, "x") == {
        "ok": True
    }


class _FakeSseResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def test_sse_response_parsing_finds_matching_message():
    lines = [
        ": comment",
        "event: message",
        'data: {"jsonrpc": "2.0", "method": "notifications/progress"}',
        "",
        "event: message",
        'data: {"jsonrpc": "2.0", "id": "req-1",',
        'data:  "result": {"tools": []}}',
        "",
    ]
    message = _HttpTransport._read_sse_response(_FakeSseResponse(lines), "req-1", "tools/list")
    assert message["result"] == {"tools": []}


def test_sse_response_without_answer_raises():
    lines = ['data: {"jsonrpc": "2.0", "method": "noise"}', ""]
    with pytest.raises(McpError, match="ended without a response"):
        _HttpTransport._read_sse_response(_FakeSseResponse(lines), "req-1", "tools/list")


def test_http_transport_rejects_redirects():
    class RedirectResponse:
        status_code = 302

        def close(self):
            pass

    class RedirectSession:
        def post(self, *args, **kwargs):
            assert kwargs["allow_redirects"] is False
            return RedirectResponse()

    transport = object.__new__(_HttpTransport)
    transport._url = "https://example.test/mcp"
    transport._session = RedirectSession()

    with pytest.raises(McpError, match="redirects are not allowed"):
        transport._post({"jsonrpc": "2.0"}, timeout=1, stream=True)
