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

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlsplit

import requests


logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "geist", "version": "0.1.0"}
_MAX_TOOL_LIST_PAGES = 25
_MAX_DISCOVERED_TOOLS = 500
_MAX_STDIO_LINE_CHARS = 1_000_000
_MAX_HTTP_RESPONSE_BYTES = 2_000_000
_MAX_SSE_EVENTS = 1_000
_MAX_TOOL_RESULT_CHARS = 100_000
_INHERITED_STDIO_ENVIRONMENT_KEYS = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
)


class McpError(RuntimeError):
    """A transport, protocol, or server-reported MCP failure."""


def _remaining_seconds(deadline: float, operation: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise McpError(f"MCP operation '{operation}' exceeded its end-to-end deadline")
    return remaining


def _filtered_child_environment(configured: dict[str, str]) -> dict[str, str]:
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _INHERITED_STDIO_ENVIRONMENT_KEYS
    }
    return {**inherited, **configured}


@dataclass(frozen=True)
class McpServerConfig:
    """Connection settings for one configured MCP server."""

    server_id: int
    name: str
    transport: str  # "stdio" | "http"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "transport": self.transport,
                "command": self.command,
                "args": list(self.args),
                "env": self.env,
                "url": self.url,
                "headers": self.headers,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
                env=_filtered_child_environment(config.env),
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
        while True:
            line = stdout.readline(_MAX_STDIO_LINE_CHARS + 1)
            if not line:
                break
            if len(line) > _MAX_STDIO_LINE_CHARS:
                logger.warning("Closing MCP stdio server after oversized response line")
                self.close()
                break
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
            self._stderr_tail.append(line[:2000].rstrip())

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

    def request(
        self,
        method: str,
        params: dict[str, Any] | None,
        timeout: float,
        cancellation: threading.Event | None = None,
    ) -> Any:
        request_id = uuid.uuid4().hex
        event = threading.Event()
        with self._pending_lock:
            self._pending_events[request_id] = event
        try:
            message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                message["params"] = params
            self._write(message)
            deadline = time.monotonic() + timeout
            while not event.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
                if cancellation is not None and cancellation.is_set():
                    raise McpError(f"MCP request '{method}' was cancelled")
                if time.monotonic() >= deadline:
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

    def notify(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> None:
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
        parsed_url = urlsplit(config.url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.fragment
        ):
            raise McpError(f"MCP server '{config.name}' has an invalid HTTP URL")
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

    def request(
        self,
        method: str,
        params: dict[str, Any] | None,
        timeout: float,
        cancellation: threading.Event | None = None,
    ) -> Any:
        if cancellation is not None and cancellation.is_set():
            raise McpError(f"MCP request '{method}' was cancelled")
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
                payload = self._read_sse_response(
                    response,
                    request_id,
                    method,
                    cancellation=cancellation,
                )
            else:
                try:
                    payload = json.loads(self._read_bounded_body(response))
                except (UnicodeDecodeError, ValueError) as error:
                    raise McpError(
                        f"MCP server returned a non-JSON response to '{method}'"
                    ) from error
        return _unwrap_response(payload, method)

    def notify(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        response = self._post(message, timeout=timeout, stream=False)
        response.close()

    def _post(self, message: dict[str, Any], timeout: float, stream: bool) -> requests.Response:
        try:
            response = self._session.post(
                self._url,
                data=json.dumps(message, ensure_ascii=False).encode("utf-8"),
                timeout=timeout,
                stream=stream,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise McpError(f"MCP server request failed: {error}") from error
        if 300 <= response.status_code < 400:
            response.close()
            raise McpError("MCP server redirects are not allowed")
        if response.status_code >= 400:
            body = self._read_bounded_body(response)[:500]
            response.close()
            raise McpError(f"MCP server returned HTTP {response.status_code}: {body}")
        return response

    @staticmethod
    def _read_sse_response(
        response: requests.Response,
        request_id: str,
        method: str,
        cancellation: threading.Event | None = None,
    ) -> dict[str, Any]:
        data_lines: list[str] = []
        total_bytes = 0
        event_count = 0
        for raw_line in response.iter_lines(decode_unicode=True):
            if cancellation is not None and cancellation.is_set():
                raise McpError(f"MCP request '{method}' was cancelled")
            if raw_line is None:
                continue
            line = raw_line.rstrip("\r") if isinstance(raw_line, str) else ""
            total_bytes += len(line.encode("utf-8")) + 1
            if total_bytes > _MAX_HTTP_RESPONSE_BYTES:
                raise McpError("MCP event stream exceeded the response size limit")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if line:
                continue  # other SSE fields (event:, id:) are irrelevant here
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            data_lines = []
            event_count += 1
            if event_count > _MAX_SSE_EVENTS:
                raise McpError("MCP event stream exceeded the event limit")
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

    @staticmethod
    def _read_bounded_body(response: requests.Response) -> str:
        chunks: list[bytes] = []
        total_bytes = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > _MAX_HTTP_RESPONSE_BYTES:
                raise McpError("MCP response exceeded the response size limit")
            chunks.append(chunk)
        return b"".join(chunks).decode(response.encoding or "utf-8")

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

    def __init__(self, config: McpServerConfig, *, deadline: float | None = None):
        self.config = config
        self._operation_lock = threading.Lock()
        deadline = deadline or (time.monotonic() + config.timeout_seconds)
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
                timeout=_remaining_seconds(deadline, "initialize"),
            )
            self.server_info = (result or {}).get("serverInfo", {})
            self.instructions = (result or {}).get("instructions")
            negotiated = (result or {}).get("protocolVersion") or PROTOCOL_VERSION
            if isinstance(self._transport, _HttpTransport):
                self._transport._session.headers["MCP-Protocol-Version"] = str(negotiated)
            self._transport.notify(
                "notifications/initialized",
                timeout=_remaining_seconds(deadline, "notifications/initialized"),
            )
        except Exception:
            self._transport.close()
            raise

    def list_tools(self, *, deadline: float | None = None) -> list[dict[str, Any]]:
        deadline = deadline or (time.monotonic() + self.config.timeout_seconds)
        with self._operation_lock:
            tools: list[dict[str, Any]] = []
            cursor: str | None = None
            for _ in range(_MAX_TOOL_LIST_PAGES):
                params: dict[str, Any] = {"cursor": cursor} if cursor else {}
                result = self._transport.request(
                    "tools/list",
                    params,
                    timeout=_remaining_seconds(deadline, "tools/list"),
                )
                page = (result or {}).get("tools") or []
                tools.extend(tool for tool in page if isinstance(tool, dict))
                if len(tools) > _MAX_DISCOVERED_TOOLS:
                    raise McpError("MCP server exceeded the discovered tool limit")
                cursor = (result or {}).get("nextCursor")
                if not cursor:
                    return tools
            raise McpError("MCP server exceeded the tool pagination limit")

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        deadline: float | None = None,
        cancellation: threading.Event | None = None,
    ) -> str:
        deadline = deadline or (time.monotonic() + self.config.timeout_seconds)
        with self._operation_lock:
            result = self._transport.request(
                "tools/call",
                {"name": name, "arguments": arguments},
                timeout=_remaining_seconds(deadline, "tools/call"),
                cancellation=cancellation,
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
        if len(content) > _MAX_TOOL_RESULT_CHARS:
            raise McpError("MCP tool result exceeded the content size limit")
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
        self._generations: dict[int, int] = {}
        self._connecting: dict[tuple[int, int, str], threading.Event] = {}
        self._closed = False

    def _connection(
        self,
        config: McpServerConfig,
        *,
        deadline: float | None = None,
    ) -> McpConnection:
        deadline = deadline or (time.monotonic() + config.timeout_seconds)
        while True:
            stale_connection: McpConnection | None = None
            with self._lock:
                if self._closed:
                    raise McpError("MCP client manager is shut down")
                generation = self._generations.get(config.server_id, 0)
                existing = self._connections.get(config.server_id)
                if existing is not None and existing.config.fingerprint == config.fingerprint:
                    return existing
                if existing is not None:
                    stale_connection = self._connections.pop(config.server_id)
                flight_key = (config.server_id, generation, config.fingerprint)
                flight = self._connecting.get(flight_key)
                leader = flight is None
                if leader:
                    flight = threading.Event()
                    self._connecting[flight_key] = flight

            if stale_connection is not None:
                stale_connection.close()
            assert flight is not None
            if not leader:
                if not flight.wait(_remaining_seconds(deadline, "connect")):
                    raise McpError("MCP connection setup timed out")
                continue

            try:
                connection = McpConnection(config, deadline=deadline)
            except Exception:
                with self._lock:
                    self._connecting.pop(flight_key, None)
                    flight.set()
                raise

            replaced: McpConnection | None = None
            retry = False
            with self._lock:
                self._connecting.pop(flight_key, None)
                if self._closed or self._generations.get(config.server_id, 0) != generation:
                    retry = not self._closed
                else:
                    current = self._connections.get(config.server_id)
                    if current is not None and current.config.fingerprint == config.fingerprint:
                        replaced = connection
                        connection = current
                    else:
                        replaced = current
                        self._connections[config.server_id] = connection
                flight.set()
            if replaced is not None:
                replaced.close()
            if self._closed:
                connection.close()
                raise McpError("MCP client manager is shut down")
            if retry:
                connection.close()
                continue
            return connection

    def list_tools(self, config: McpServerConfig) -> list[dict[str, Any]]:
        deadline = time.monotonic() + config.timeout_seconds
        try:
            return self._connection(config, deadline=deadline).list_tools(deadline=deadline)
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
        cancellation: threading.Event | None = None,
    ) -> str:
        deadline = time.monotonic() + config.timeout_seconds
        try:
            return self._connection(config, deadline=deadline).call_tool(
                name,
                arguments,
                deadline=deadline,
                cancellation=cancellation,
            )
        except McpError:
            self.invalidate(config.server_id)
            raise
        except Exception as error:
            self.invalidate(config.server_id)
            raise McpError(f"MCP server '{config.name}' failed: {error}") from error

    def invalidate(self, server_id: int) -> None:
        with self._lock:
            self._generations[server_id] = self._generations.get(server_id, 0) + 1
            connection = self._connections.pop(server_id, None)
        if connection is not None:
            connection.close()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            connections = list(self._connections.values())
            self._connections.clear()
            server_ids = set(self._generations)
            server_ids.update(key[0] for key in self._connecting)
            for server_id in server_ids:
                self._generations[server_id] = self._generations.get(server_id, 0) + 1
        for connection in connections:
            connection.close()
