"""
API endpoints for configuring MCP (Model Context Protocol) servers.

Configured servers are persisted per user; only servers explicitly marked
enabled contribute tools to chat via the registry's MCP tool source.
"""
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.database.geist_user import get_default_user
from app.models.database.mcp_server import (
    McpServerModel,
    create_mcp_server,
    delete_mcp_server,
    get_mcp_server,
    list_mcp_servers,
    update_mcp_server,
)
from app.services.mcp_client import McpError
from app.services.mcp_tool_source import (
    config_from_model,
    get_mcp_manager,
    get_mcp_tool_source,
)


logger = logging.getLogger(__name__)

router = APIRouter()

_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


def get_current_user():
    """Get current user (placeholder - should integrate with actual auth system)."""
    return get_default_user()


def _validate_transport_fields(transport: str, command: str | None, url: str | None) -> None:
    if transport == "stdio":
        if not (command or "").strip():
            raise ValueError("stdio servers require a command")
    elif transport == "http" and (
        not url or not url.lower().startswith(("http://", "https://"))
    ):
        raise ValueError("http servers require an http(s) URL")


class McpServerCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_NAME_PATTERN)
    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = Field(default=None, max_length=1024)
    args: list[str] = Field(default_factory=list, max_length=64)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = Field(default=None, max_length=2048)
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = False
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)

    @model_validator(mode="after")
    def _check_transport(self) -> "McpServerCreate":
        _validate_transport_fields(self.transport, self.command, self.url)
        return self


class McpServerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, pattern=_NAME_PATTERN)
    transport: Literal["stdio", "http"] | None = None
    command: str | None = Field(default=None, max_length=1024)
    args: list[str] | None = Field(default=None, max_length=64)
    env: dict[str, str] | None = None
    url: str | None = Field(default=None, max_length=2048)
    headers: dict[str, str] | None = None
    enabled: bool | None = None
    timeout_seconds: float | None = Field(default=None, ge=1.0, le=600.0)


class McpServerResponse(BaseModel):
    mcp_server_id: int
    user_id: int
    name: str
    transport: str
    command: str | None
    args: list[str]
    env: dict[str, str]
    url: str | None
    headers: dict[str, str]
    enabled: bool
    timeout_seconds: float
    create_date: Any
    update_date: Any


class McpToolInfo(BaseModel):
    name: str
    description: str = ""


class McpServerTestResponse(BaseModel):
    ok: bool
    error: str | None = None
    tools: list[McpToolInfo] = Field(default_factory=list)


def _response(server: McpServerModel) -> McpServerResponse:
    return McpServerResponse(**server.__dict__)


def _invalidate(server_id: int | None = None) -> None:
    if server_id is not None:
        get_mcp_manager().invalidate(server_id)
    get_mcp_tool_source().invalidate()


@router.get("/servers", response_model=list[McpServerResponse])
async def list_servers():
    try:
        user = get_current_user()
        return [_response(server) for server in list_mcp_servers(user.user_id)]
    except Exception as e:
        logger.error(f"Error listing MCP servers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/servers", response_model=McpServerResponse, status_code=201)
async def create_server(request: McpServerCreate):
    try:
        user = get_current_user()
        if any(server.name == request.name for server in list_mcp_servers(user.user_id)):
            raise HTTPException(
                status_code=409, detail=f"An MCP server named '{request.name}' already exists"
            )
        server = create_mcp_server(user.user_id, request.model_dump())
        _invalidate()
        return _response(server)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating MCP server: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/servers/{mcp_server_id}", response_model=McpServerResponse)
async def get_server(mcp_server_id: int):
    server = get_mcp_server(mcp_server_id)
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return _response(server)


@router.put("/servers/{mcp_server_id}", response_model=McpServerResponse)
async def update_server(mcp_server_id: int, request: McpServerUpdate):
    try:
        existing = get_mcp_server(mcp_server_id)
        if not existing:
            raise HTTPException(status_code=404, detail="MCP server not found")
        updates = request.model_dump(exclude_unset=True)

        merged_transport = updates.get("transport", existing.transport)
        merged_command = updates.get("command", existing.command)
        merged_url = updates.get("url", existing.url)
        try:
            _validate_transport_fields(merged_transport, merged_command, merged_url)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        new_name = updates.get("name")
        if new_name and new_name != existing.name and any(
            server.name == new_name for server in list_mcp_servers(existing.user_id)
        ):
            raise HTTPException(
                status_code=409, detail=f"An MCP server named '{new_name}' already exists"
            )

        server = update_mcp_server(mcp_server_id, updates)
        if not server:
            raise HTTPException(status_code=404, detail="MCP server not found")
        _invalidate(mcp_server_id)
        return _response(server)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating MCP server {mcp_server_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.delete("/servers/{mcp_server_id}", status_code=204)
async def delete_server(mcp_server_id: int):
    try:
        if not delete_mcp_server(mcp_server_id):
            raise HTTPException(status_code=404, detail="MCP server not found")
        _invalidate(mcp_server_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting MCP server {mcp_server_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/servers/{mcp_server_id}/test", response_model=McpServerTestResponse)
async def test_server(mcp_server_id: int):
    """Connect to the server and list its tools without enabling it."""
    server = get_mcp_server(mcp_server_id)
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    try:
        tools = get_mcp_manager().list_tools(config_from_model(server))
    except McpError as error:
        return McpServerTestResponse(ok=False, error=str(error))
    except Exception as e:
        logger.error(f"Unexpected error testing MCP server {mcp_server_id}: {e}")
        return McpServerTestResponse(ok=False, error="Unexpected connection failure")
    return McpServerTestResponse(
        ok=True,
        tools=[
            McpToolInfo(
                name=str(tool.get("name", "")),
                description=str(tool.get("description") or ""),
            )
            for tool in tools
        ],
    )


@router.get("/tools", response_model=list[McpToolInfo])
async def list_mounted_tools():
    """List the MCP tools currently mounted into the chat tool registry."""
    try:
        return [
            McpToolInfo(name=definition.name, description=definition.description)
            for definition in get_mcp_tool_source().definitions()
        ]
    except Exception as e:
        logger.error(f"Error listing mounted MCP tools: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
