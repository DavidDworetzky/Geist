"""Minimal synchronous MCP (Model Context Protocol) client.

Speaks JSON-RPC 2.0 over the two standard MCP transports:

- ``stdio``: a locally spawned server process, newline-delimited JSON-RPC.
- ``http``: Streamable HTTP; each request is a POST that may answer with a
  plain JSON body or a text/event-stream containing the response message.

Only the client surface Geist needs is implemented — ``initialize``,
``tools/list`` (paginated), and ``tools/call``. Server-initiated requests
(sampling, elicitation) are answered with JSON-RPC "method not found" so
well-behaved servers degrade gracefully. Like the provider clients in
``agents.online_agent``, this intentionally avoids an SDK dependency.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, cast

import requests


logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "geist", "version": "0.1.0"}
_MAX_TOOL_LIST_PAGES = 25


class McpError(RuntimeError):
    """A transport, protocol, or server-reported MCP failure."""


@dataclass(frozen=True)
class McpServerConfig:
    """Connection settings for one configured MCP server."""

    server_id: int
    name: str
    transport: str  # "stdio" | "http"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    working_directory: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    connector_kind: str = "custom"
    trusted: bool = False
    recipient_allowlist: tuple[str, ...] = ()
    max_writes_per_hour: int = 20

    @property
    def fingerprint(self) -> str:
        return json.dumps(
            {
                "transport": self.transport,
                "command": self.command,
                "args": list(self.args),
                "env": self.env,
                "working_directory": self.working_directory,
                "url": self.url,
                "headers": self.headers,
            },
            sort_keys=True,
        )


class _StdioTransport:
    """Newline-delimited JSON-RPC over a spawned server process."""

    def __init__(self, config: McpServerConfig):
        if not config.command:
            raise McpError(f"MCP server '{config.name}' has no command configured")
        try:
            self._process = subprocess.Popen(
                [config.command, *config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, **config.env},
                cwd=config.working_directory,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            raise McpError(f"Could not start MCP server '{config.name}': {error}") from error
        self._write_lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._pending_events: dict[str, threading.Event] = {}
        self._pending_lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=20)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        for line in stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Discarding non-JSON MCP stdio line: %.200s", line)
                continue
            self._handle_message(message)
        # EOF: wake every waiter so requests fail fast instead of timing out.
        with self._pending_lock:
            for event in self._pending_events.values():
                event.set()

    def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        for line in stderr:
            self._stderr_tail.append(line.rstrip())

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_id = message.get("id")
        if message_id is not None and "method" in message:
            # Server-initiated request (sampling/elicitation): decline politely.
            self._write(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32601, "message": "Method not supported by this client"},
                }
            )
            return
        if message_id is None:
            return  # notification
        key = str(message_id)
        with self._pending_lock:
            event = self._pending_events.get(key)
            if event is not None:
                self._pending[key] = message
                event.set()

    def _write(self, message: dict[str, Any]) -> None:
        stdin = self._process.stdin
        if stdin is None or self._process.poll() is not None:
            raise McpError(f"MCP server process exited: {self._stderr_summary()}")
        payload = json.dumps(message, ensure_ascii=False)
        with self._write_lock:
            try:
                stdin.write(payload + "\n")
                stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise McpError(f"MCP server pipe closed: {self._stderr_summary()}") from error

    def _stderr_summary(self) -> str:
        return " | ".join(self._stderr_tail) or "no stderr output"

    def request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        request_id = uuid.uuid4().hex
        event = threading.Event()
        with self._pending_lock:
            self._pending_events[request_id] = event
        try:
            message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                message["params"] = params
            self._write(message)
            if not event.wait(timeout):
                raise McpError(f"MCP request '{method}' timed out after {timeout:g}s")
            with self._pending_lock:
                response = self._pending.pop(request_id, None)
            if response is None:
                raise McpError(
                    f"MCP server closed the connection during '{method}': "
                    f"{self._stderr_summary()}"
                )
            return _unwrap_response(response, method)
        finally:
            with self._pending_lock:
                self._pending_events.pop(request_id, None)
                self._pending.pop(request_id, None)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._write(message)

    def close(self) -> None:
        try:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
        except OSError:
            logger.warning("Could not terminate MCP server process", exc_info=True)


class _HttpTransport:
    """Streamable HTTP: JSON-RPC over POST with JSON or SSE responses."""

    def __init__(self, config: McpServerConfig):
        if not config.url:
            raise McpError(f"MCP server '{config.name}' has no URL configured")
        self._url = config.url
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                **config.headers,
            }
        )
        self._mcp_session_id: str | None = None

    def request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        request_id = uuid.uuid4().hex
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        response = self._post(message, timeout, stream=True)
        with response:
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                self._mcp_session_id = session_id
                self._session.headers["Mcp-Session-Id"] = session_id
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()
            if content_type == "text/event-stream":
                payload = self._read_sse_response(response, request_id, method)
            else:
                try:
                    payload = response.json()
                except ValueError as error:
                    raise McpError(
                        f"MCP server returned a non-JSON response to '{method}'"
                    ) from error
        return _unwrap_response(payload, method)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        response = self._post(message, timeout=10, stream=False)
        response.close()

    def _post(self, message: dict[str, Any], timeout: float, stream: bool) -> requests.Response:
        try:
            response = self._session.post(
                self._url,
                data=json.dumps(message, ensure_ascii=False).encode("utf-8"),
                timeout=timeout,
                stream=stream,
            )
        except requests.RequestException as error:
            raise McpError(f"MCP server request failed: {error}") from error
        if response.status_code >= 400:
            body = response.text[:500]
            response.close()
            raise McpError(f"MCP server returned HTTP {response.status_code}: {body}")
        return response

    @staticmethod
    def _read_sse_response(
        response: requests.Response, request_id: str, method: str
    ) -> dict[str, Any]:
        data_lines: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.rstrip("\r") if isinstance(raw_line, str) else ""
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if line:
                continue  # other SSE fields (event:, id:) are irrelevant here
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            data_lines = []
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(message, dict)
                and message.get("id") == request_id
                and ("result" in message or "error" in message)
            ):
                return cast(dict[str, Any], message)
        raise McpError(f"MCP event stream ended without a response to '{method}'")

    def close(self) -> None:
        self._session.close()


def _unwrap_response(message: Any, method: str) -> Any:
    if not isinstance(message, dict):
        raise McpError(f"MCP server returned an invalid response to '{method}'")
    if "error" in message:
        error = message["error"] or {}
        raise McpError(
            f"MCP server error for '{method}': "
            f"{error.get('message', 'unknown error')} (code {error.get('code')})"
        )
    return message.get("result")


class McpConnection:
    """One initialized MCP session over either transport."""

    def __init__(self, config: McpServerConfig):
        self.config = config
        if config.transport == "stdio":
            self._transport: _StdioTransport | _HttpTransport = _StdioTransport(config)
        elif config.transport == "http":
            self._transport = _HttpTransport(config)
        else:
            raise McpError(f"Unknown MCP transport '{config.transport}'")
        try:
            result = self._transport.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
                timeout=config.timeout_seconds,
            )
            self.server_info = (result or {}).get("serverInfo", {})
            self.instructions = (result or {}).get("instructions")
            negotiated = (result or {}).get("protocolVersion") or PROTOCOL_VERSION
            if isinstance(self._transport, _HttpTransport):
                self._transport._session.headers["MCP-Protocol-Version"] = str(negotiated)
            self._transport.notify("notifications/initialized")
        except Exception:
            self._transport.close()
            raise

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(_MAX_TOOL_LIST_PAGES):
            params: dict[str, Any] = {"cursor": cursor} if cursor else {}
            result = self._transport.request(
                "tools/list", params, timeout=self.config.timeout_seconds
            )
            page = (result or {}).get("tools") or []
            tools.extend(tool for tool in page if isinstance(tool, dict))
            cursor = (result or {}).get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> str:
        params: dict[str, Any] = {"name": name, "arguments": arguments}
        if idempotency_key:
            params["_meta"] = {"geist/idempotencyKey": idempotency_key}
        result = self._transport.request(
            "tools/call",
            params,
            timeout=self.config.timeout_seconds,
        )
        result = result or {}
        parts: list[str] = []
        for item in result.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(f"[unsupported {item.get('type', 'unknown')} content omitted]")
        if not parts and result.get("structuredContent") is not None:
            parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
        content = "\n".join(part for part in parts if part)
        if result.get("isError"):
            raise McpError(content or f"MCP tool '{name}' reported an error")
        return content

    def close(self) -> None:
        self._transport.close()


class McpClientManager:
    """Caches one live connection per configured server, reconnecting on change."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connections: dict[int, McpConnection] = {}
        self._idempotency_lock = threading.Lock()
        self._idempotency_results: OrderedDict[str, str] = OrderedDict()

    def _connection(self, config: McpServerConfig) -> McpConnection:
        with self._lock:
            existing = self._connections.get(config.server_id)
            if existing is not None and existing.config.fingerprint == config.fingerprint:
                return existing
            if existing is not None:
                existing.close()
                del self._connections[config.server_id]
        connection = McpConnection(config)
        with self._lock:
            current = self._connections.get(config.server_id)
            if current is not None and current.config.fingerprint == config.fingerprint:
                connection.close()
                return current
            if current is not None:
                current.close()
            self._connections[config.server_id] = connection
        return connection

    def list_tools(self, config: McpServerConfig) -> list[dict[str, Any]]:
        try:
            return self._connection(config).list_tools()
        except McpError:
            self.invalidate(config.server_id)
            raise
        except Exception as error:
            self.invalidate(config.server_id)
            raise McpError(f"MCP server '{config.name}' failed: {error}") from error

    def call_tool(
        self,
        config: McpServerConfig,
        name: str,
        arguments: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> str:
        if idempotency_key:
            with self._idempotency_lock:
                cached = self._idempotency_results.get(idempotency_key)
                if cached is not None:
                    self._idempotency_results.move_to_end(idempotency_key)
                    return cached
        try:
            content = self._connection(config).call_tool(
                name,
                arguments,
                idempotency_key=idempotency_key,
            )
            if idempotency_key:
                with self._idempotency_lock:
                    self._idempotency_results[idempotency_key] = content
                    while len(self._idempotency_results) > 1000:
                        self._idempotency_results.popitem(last=False)
            return content
        except McpError:
            self.invalidate(config.server_id)
            raise
        except Exception as error:
            self.invalidate(config.server_id)
            raise McpError(f"MCP server '{config.name}' failed: {error}") from error

    def invalidate(self, server_id: int) -> None:
        with self._lock:
            connection = self._connections.pop(server_id, None)
        if connection is not None:
            connection.close()

    def shutdown(self) -> None:
        with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
        with self._idempotency_lock:
            self._idempotency_results.clear()
        for connection in connections:
            connection.close()
