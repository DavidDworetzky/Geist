"""Mount enabled MCP servers' tools into the unified chat ToolRegistry.

Each enabled server contributes its tools as ``mcp.<server>.<tool>``
definitions with the server-provided JSON input schema. Discovery results are
cached briefly so chat turns do not hammer servers; configuration changes
invalidate the cache immediately. A failing server logs and contributes no
tools instead of breaking chat.
"""

from __future__ import annotations

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
        url=server.url,
        headers=dict(server.headers or {}),
        timeout_seconds=server.timeout_seconds,
    )


class McpToolSource:
    """ToolSource serving definitions discovered from enabled MCP servers."""

    name = "mcp"

    def __init__(
        self,
        manager: McpClientManager,
        servers_loader: Callable[[], list[McpServerModel]] = list_enabled_mcp_servers,
        cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS,
    ):
        self._manager = manager
        self._servers_loader = servers_loader
        self._cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.Lock()
        self._cached: list[ToolDefinition] | None = None
        self._cache_expires_at = 0.0

    def invalidate(self) -> None:
        with self._lock:
            self._cached = None
            self._cache_expires_at = 0.0

    def definitions(self) -> list[ToolDefinition]:
        with self._lock:
            if self._cached is not None and time.monotonic() < self._cache_expires_at:
                return list(self._cached)

        definitions: list[ToolDefinition] = []
        try:
            servers = self._servers_loader()
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
            self._cached = list(definitions)
            self._cache_expires_at = time.monotonic() + self._cache_ttl_seconds
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
            content = self._manager.call_tool(_config, _tool, arguments)
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
