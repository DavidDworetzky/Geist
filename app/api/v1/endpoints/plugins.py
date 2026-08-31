"""
API endpoints for installed agent plugins (agent-plugins.org format).

Plugins are read from the plugin directory on disk; this API lists what was
discovered and re-scans on demand. Installation is a filesystem operation
(drop a plugin directory into the plugin root), so there are no create or
delete routes.
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.security.operator import (
    OperatorCapability,
    OperatorPrincipal,
    require_operator_capability,
)
from app.services.plugin_context import get_plugin_mcp_source
from app.services.plugin_loader import enabled_plugin_names, get_plugin_registry


logger = logging.getLogger(__name__)

router = APIRouter()
_require_tools_operator = require_operator_capability(OperatorCapability.TOOLS_MANAGE)


class PluginSkillResponse(BaseModel):
    name: str
    qualified_name: str
    description: str


class PluginMcpServerResponse(BaseModel):
    name: str
    qualified_name: str
    transport: str
    enabled: bool


class PluginResponse(BaseModel):
    name: str
    version: str | None
    description: str | None
    enabled_for_mcp: bool
    skills: list[PluginSkillResponse]
    mcp_servers: list[PluginMcpServerResponse]


class PluginListResponse(BaseModel):
    plugin_dir: str
    plugins: list[PluginResponse]


def _plugin_list() -> PluginListResponse:
    registry = get_plugin_registry()
    enabled = enabled_plugin_names()
    plugins = []
    for plugin in registry.plugins():
        plugin_enabled = plugin.name in enabled
        plugins.append(
            PluginResponse(
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                enabled_for_mcp=plugin_enabled,
                skills=[
                    PluginSkillResponse(
                        name=skill.name,
                        qualified_name=skill.qualified_name,
                        description=skill.description,
                    )
                    for skill in plugin.skills
                ],
                mcp_servers=[
                    PluginMcpServerResponse(
                        name=server.name,
                        qualified_name=server.qualified_name,
                        transport=server.transport,
                        enabled=plugin_enabled,
                    )
                    for server in plugin.mcp_servers
                ],
            )
        )
    return PluginListResponse(plugin_dir=str(registry.plugin_dir), plugins=plugins)


@router.get("", response_model=PluginListResponse)
@router.get("/", response_model=PluginListResponse, include_in_schema=False)
def list_plugins(
    _operator: OperatorPrincipal = Depends(_require_tools_operator),
) -> PluginListResponse:
    return _plugin_list()


@router.post("/refresh", response_model=PluginListResponse)
def refresh_plugins(
    _operator: OperatorPrincipal = Depends(_require_tools_operator),
) -> PluginListResponse:
    """Re-scan the plugin directory and invalidate cached MCP discovery."""
    get_plugin_registry().refresh()
    source = get_plugin_mcp_source()
    if source is not None:
        source.invalidate()
    return _plugin_list()
