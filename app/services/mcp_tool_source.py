"""Mount enabled MCP servers' tools into the unified chat ToolRegistry.

Each enabled server contributes its tools as ``mcp.<server>.<tool>``
definitions with the server-provided JSON input schema. Discovery results are
cached briefly so chat turns do not hammer servers; configuration changes
invalidate the cache immediately. A failing server logs and contributes no
tools instead of breaking chat.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable

from agents.models.tool_calling import ToolContext, ToolDefinition, ToolExecutionOutput
from app.models.database.mcp_server import McpServerModel, list_enabled_mcp_servers
from app.services.mcp_client import McpClientManager, McpError, McpServerConfig


logger = logging.getLogger(__name__)

_DEFAULT_CACHE_TTL_SECONDS = 60.0


def config_from_model(server: McpServerModel) -> McpServerConfig:
    return McpServerConfig(
        server_id=server.mcp_server_id,
        name=server.name,
        transport=server.transport,
        command=server.command,
        args=tuple(server.args or []),
        env=dict(server.env or {}),
        working_directory=server.working_directory,
        url=server.url,
        headers=dict(server.headers or {}),
        timeout_seconds=server.timeout_seconds,
        connector_kind=server.connector_kind,
        trusted=server.trusted,
        recipient_allowlist=tuple(server.recipient_allowlist or []),
        max_writes_per_hour=server.max_writes_per_hour,
    )


def _idempotency_key(
    context: ToolContext,
    config: McpServerConfig,
    tool_name: str,
    arguments: dict,
) -> str:
    payload = json.dumps(
        {
            "user_id": context.user_id,
            "run_id": context.run_id,
            "server_id": config.server_id,
            "tool": tool_name,
            "arguments": arguments,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class McpToolSource:
    """ToolSource serving definitions discovered from enabled MCP servers."""

    name = "mcp"

    def __init__(
        self,
        manager: McpClientManager,
        servers_loader: Callable[[int | None], list[McpServerModel]] = list_enabled_mcp_servers,
        cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS,
    ):
        self._manager = manager
        self._servers_loader = servers_loader
        self._cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.Lock()
        self._cached: dict[int | None, tuple[list[ToolDefinition], float]] = {}

    def invalidate(self) -> None:
        with self._lock:
            self._cached.clear()

    def definitions(self, context: ToolContext | None = None) -> list[ToolDefinition]:
        user_id = context.user_id if context is not None else None
        with self._lock:
            cached = self._cached.get(user_id)
            if cached is not None and time.monotonic() < cached[1]:
                return list(cached[0])

        definitions: list[ToolDefinition] = []
        try:
            servers = self._servers_loader(user_id)
        except Exception:
            logger.exception("Could not load MCP server configurations")
            servers = []
        for server in servers:
            config = config_from_model(server)
            try:
                tools = self._manager.list_tools(config)
            except McpError as error:
                logger.warning("Skipping MCP server '%s': %s", server.name, error)
                continue
            for tool in tools:
                definition = self._definition(config, tool)
                if definition is not None:
                    definitions.append(definition)

        with self._lock:
            self._cached[user_id] = (
                list(definitions),
                time.monotonic() + self._cache_ttl_seconds,
            )
        return definitions

    def _definition(self, config: McpServerConfig, tool: dict) -> ToolDefinition | None:
        name_value = tool.get("name")
        if not isinstance(name_value, str) or not name_value:
            logger.warning("MCP server '%s' returned a tool without a name", config.name)
            return None
        tool_name: str = name_value
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        description = tool.get("description") or tool_name

        def handler(
            context: ToolContext,
            arguments: dict,
            *,
            _config: McpServerConfig = config,
            _tool: str = tool_name,
        ) -> ToolExecutionOutput:
            content = self._manager.call_tool(
                _config,
                _tool,
                arguments,
                idempotency_key=_idempotency_key(context, _config, _tool, arguments),
            )
            return ToolExecutionOutput(
                content=content or "(empty tool result)",
                summary=f"Called {_tool} on MCP server {_config.name}",
            )

        return ToolDefinition(
            name=f"mcp.{config.name}.{tool_name}",
            description=f"[MCP: {config.name}] {description}",
            arguments_schema=schema,
            handler=handler,
            side_effect="external_write",
            # MCP does not expose a trustworthy read/write classification for
            # arbitrary server tools. Fail closed and require a user decision
            # before dispatching anything discovered at runtime.
            requires_approval=True,
            timeout_seconds=config.timeout_seconds + 5.0,
            source_adapter=f"mcp:{config.name}",
            # Email remains untrusted even after an operator promotes its
            # server. Trust promotion only affects non-email MCP servers.
            untrusted_external=(not config.trusted or config.connector_kind != "custom"),
            always_untrusted_content=config.connector_kind != "custom",
            security_source_id=config.server_id,
            recipient_allowlist=config.recipient_allowlist,
            max_writes_per_hour=config.max_writes_per_hour,
        )


_manager: McpClientManager | None = None
_source: McpToolSource | None = None
_singleton_lock = threading.Lock()


def get_mcp_manager() -> McpClientManager:
    global _manager
    with _singleton_lock:
        if _manager is None:
            _manager = McpClientManager()
        return _manager


def get_mcp_tool_source() -> McpToolSource:
    global _source
    manager = get_mcp_manager()
    with _singleton_lock:
        if _source is None:
            _source = McpToolSource(manager)
        return _source
