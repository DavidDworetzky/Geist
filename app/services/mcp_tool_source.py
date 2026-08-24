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
from concurrent.futures import ThreadPoolExecutor

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
        servers_loader: Callable[[int | None], list[McpServerModel]] = list_enabled_mcp_servers,
        cache_ttl_seconds: float = _DEFAULT_CACHE_TTL_SECONDS,
    ):
        self._manager = manager
        self._servers_loader = servers_loader
        self._cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.Lock()
        self._generation = 0
        self._cached: dict[int | None, tuple[list[ToolDefinition], float, int]] = {}
        self._inflight: dict[tuple[int | None, int], threading.Event] = {}

    def invalidate(self) -> None:
        with self._lock:
            self._generation += 1
            self._cached.clear()

    def definitions(self, context: ToolContext | None = None) -> list[ToolDefinition]:
        user_id = context.user_id if context is not None else None
        while True:
            with self._lock:
                generation = self._generation
                cached = self._cached.get(user_id)
                if (
                    cached is not None
                    and cached[2] == generation
                    and time.monotonic() < cached[1]
                ):
                    return list(cached[0])
                flight_key = (user_id, generation)
                flight = self._inflight.get(flight_key)
                leader = flight is None
                if leader:
                    flight = threading.Event()
                    self._inflight[flight_key] = flight
            assert flight is not None
            if leader:
                break
            flight.wait()

        try:
            definitions = self._discover_definitions(user_id)
        except Exception:
            logger.exception("Could not discover MCP tool definitions")
            definitions = []
        with self._lock:
            self._inflight.pop(flight_key, None)
            if generation == self._generation:
                self._cached[user_id] = (
                    list(definitions),
                    time.monotonic() + self._cache_ttl_seconds,
                    generation,
                )
            flight.set()
        return definitions

    def _discover_definitions(self, user_id: int | None) -> list[ToolDefinition]:
        try:
            servers = self._servers_loader(user_id)
        except Exception:
            logger.exception("Could not load MCP server configurations")
            return []

        def discover(server: McpServerModel) -> list[ToolDefinition]:
            config = config_from_model(server)
            try:
                tools = self._manager.list_tools(config)
            except McpError as error:
                logger.warning("Skipping MCP server '%s': %s", server.name, error)
                return []
            server_definitions: list[ToolDefinition] = []
            for tool in tools:
                definition = self._definition(config, tool)
                if definition is not None:
                    server_definitions.append(definition)
            return server_definitions

        if not servers:
            return []
        with ThreadPoolExecutor(
            max_workers=min(4, len(servers)),
            thread_name_prefix="geist-mcp-discovery",
        ) as executor:
            discovered = executor.map(discover, servers)
            definitions = [definition for group in discovered for definition in group]
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
                cancellation=context.cancellation,
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
            source_revision=config.fingerprint,
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
