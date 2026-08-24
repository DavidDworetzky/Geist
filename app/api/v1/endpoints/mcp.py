"""
API endpoints for configuring MCP (Model Context Protocol) servers.

Configured servers are persisted per user; only servers explicitly marked
enabled contribute tools to chat via the registry's MCP tool source.
"""

import logging
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from agents.models.tool_calling import ToolContext
from app.models.database.mcp_server import (
    McpServerModel,
    create_mcp_server,
    delete_mcp_server,
    get_mcp_server,
    list_mcp_servers,
    update_mcp_server,
)
from app.security.operator import (
    OperatorCapability,
    OperatorPrincipal,
    require_operator_capability,
)
from app.services.mcp_client import McpError
from app.services.mcp_tool_source import (
    config_from_model,
    get_mcp_manager,
    get_mcp_tool_source,
)


logger = logging.getLogger(__name__)

router = APIRouter()
_require_tools_operator = require_operator_capability(OperatorCapability.TOOLS_MANAGE)

_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_REDACTED = "<redacted>"
_HTTP_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def _validate_transport_fields(
    transport: str,
    command: str | None,
    url: str | None,
    *,
    allow_stdio: bool = True,
) -> None:
    if transport == "stdio":
        if not allow_stdio:
            raise ValueError("stdio MCP servers can be configured only by a local operator")
        if not (command or "").strip():
            raise ValueError("stdio servers require a command")
    elif transport == "http" and (not url or not url.lower().startswith(("http://", "https://"))):
        raise ValueError("http servers require an http(s) URL")


def _validate_configuration_values(
    args: list[str] | None,
    env: dict[str, str] | None,
    headers: dict[str, str] | None,
) -> None:
    if args is not None and any(len(argument) > 4096 or "\x00" in argument for argument in args):
        raise ValueError("MCP command arguments must be at most 4096 characters without NUL")
    if env is not None:
        for key, value in env.items():
            if not key or len(key) > 256 or "=" in key or "\x00" in key:
                raise ValueError("MCP environment variable names are invalid")
            if len(value) > 8192 or "\x00" in value:
                raise ValueError("MCP environment variable values are too large")
    if headers is not None:
        for key, value in headers.items():
            if len(key) > 256 or _HTTP_HEADER_NAME_PATTERN.fullmatch(key) is None:
                raise ValueError("MCP HTTP header names are invalid")
            if len(value) > 8192 or "\r" in value or "\n" in value:
                raise ValueError("MCP HTTP header values are invalid")


class McpServerCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_NAME_PATTERN)
    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = Field(default=None, max_length=1024)
    args: list[str] = Field(default_factory=list, max_length=64)
    env: dict[str, str] = Field(default_factory=dict, max_length=64)
    url: str | None = Field(default=None, max_length=2048)
    headers: dict[str, str] = Field(default_factory=dict, max_length=64)
    enabled: bool = False
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)

    @model_validator(mode="after")
    def _check_transport(self) -> "McpServerCreate":
        _validate_transport_fields(self.transport, self.command, self.url)
        _validate_configuration_values(self.args, self.env, self.headers)
        return self


class McpServerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, pattern=_NAME_PATTERN)
    transport: Literal["stdio", "http"] | None = None
    command: str | None = Field(default=None, max_length=1024)
    args: list[str] | None = Field(default=None, max_length=64)
    env: dict[str, str] | None = Field(default=None, max_length=64)
    url: str | None = Field(default=None, max_length=2048)
    headers: dict[str, str] | None = Field(default=None, max_length=64)
    enabled: bool | None = None
    timeout_seconds: float | None = Field(default=None, ge=1.0, le=600.0)

    @model_validator(mode="after")
    def _check_configuration_values(self) -> "McpServerUpdate":
        _validate_configuration_values(self.args, self.env, self.headers)
        return self


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
    values = dict(server.__dict__)
    values["env"] = {key: _REDACTED for key in server.env}
    values["headers"] = {key: _REDACTED for key in server.headers}
    return McpServerResponse(**values)


def _owned_server_or_404(mcp_server_id: int, user_id: int) -> McpServerModel:
    server = get_mcp_server(mcp_server_id)
    if server is None or server.user_id != user_id:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return server


def _preserve_redacted_secrets(
    updates: dict[str, Any],
    existing: McpServerModel,
) -> None:
    for field in ("env", "headers"):
        submitted = updates.get(field)
        if not isinstance(submitted, dict):
            continue
        stored = getattr(existing, field)
        resolved: dict[str, str] = {}
        for key, value in submitted.items():
            if value == _REDACTED:
                if key not in stored:
                    raise ValueError(f"Cannot preserve unknown redacted {field} key '{key}'")
                resolved[key] = stored[key]
            else:
                resolved[key] = value
        updates[field] = resolved


def _invalidate(server_id: int | None = None) -> None:
    if server_id is not None:
        get_mcp_manager().invalidate(server_id)
    get_mcp_tool_source().invalidate()


@router.get("/servers", response_model=list[McpServerResponse])
async def list_servers(
    operator: OperatorPrincipal = Depends(_require_tools_operator),
):
    try:
        return [_response(server) for server in list_mcp_servers(operator.user_id)]
    except Exception as e:
        logger.error(f"Error listing MCP servers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/servers", response_model=McpServerResponse, status_code=201)
async def create_server(
    request: McpServerCreate,
    operator: OperatorPrincipal = Depends(_require_tools_operator),
):
    try:
        try:
            _validate_transport_fields(
                request.transport,
                request.command,
                request.url,
                allow_stdio=operator.is_local_operator,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if any(server.name == request.name for server in list_mcp_servers(operator.user_id)):
            raise HTTPException(
                status_code=409, detail=f"An MCP server named '{request.name}' already exists"
            )
        server = create_mcp_server(operator.user_id, request.model_dump())
        _invalidate()
        return _response(server)
    except HTTPException:
        raise
    except IntegrityError as error:
        raise HTTPException(status_code=409, detail="MCP server name already exists") from error
    except Exception as e:
        logger.error(f"Error creating MCP server: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/servers/{mcp_server_id}", response_model=McpServerResponse)
async def get_server(
    mcp_server_id: int,
    operator: OperatorPrincipal = Depends(_require_tools_operator),
):
    server = _owned_server_or_404(mcp_server_id, operator.user_id)
    return _response(server)


@router.put("/servers/{mcp_server_id}", response_model=McpServerResponse)
async def update_server(
    mcp_server_id: int,
    request: McpServerUpdate,
    operator: OperatorPrincipal = Depends(_require_tools_operator),
):
    try:
        existing = _owned_server_or_404(mcp_server_id, operator.user_id)
        updates = request.model_dump(exclude_unset=True)
        try:
            _preserve_redacted_secrets(updates, existing)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        merged_transport = updates.get("transport", existing.transport)
        merged_command = updates.get("command", existing.command)
        merged_url = updates.get("url", existing.url)
        try:
            _validate_transport_fields(
                merged_transport,
                merged_command,
                merged_url,
                allow_stdio=operator.is_local_operator,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        new_name = updates.get("name")
        if (
            new_name
            and new_name != existing.name
            and any(server.name == new_name for server in list_mcp_servers(existing.user_id))
        ):
            raise HTTPException(
                status_code=409, detail=f"An MCP server named '{new_name}' already exists"
            )

        server = update_mcp_server(
            mcp_server_id,
            updates,
            user_id=operator.user_id,
        )
        if not server:
            raise HTTPException(status_code=404, detail="MCP server not found")
        _invalidate(mcp_server_id)
        return _response(server)
    except HTTPException:
        raise
    except IntegrityError as error:
        raise HTTPException(status_code=409, detail="MCP server name already exists") from error
    except Exception as e:
        logger.error(f"Error updating MCP server {mcp_server_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.delete("/servers/{mcp_server_id}", status_code=204)
async def delete_server(
    mcp_server_id: int,
    operator: OperatorPrincipal = Depends(_require_tools_operator),
):
    try:
        _owned_server_or_404(mcp_server_id, operator.user_id)
        if not delete_mcp_server(mcp_server_id, user_id=operator.user_id):
            raise HTTPException(status_code=404, detail="MCP server not found")
        _invalidate(mcp_server_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting MCP server {mcp_server_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/servers/{mcp_server_id}/test", response_model=McpServerTestResponse)
async def test_server(
    mcp_server_id: int,
    operator: OperatorPrincipal = Depends(_require_tools_operator),
):
    """Connect to the server and list its tools without enabling it."""
    server = _owned_server_or_404(mcp_server_id, operator.user_id)
    try:
        tools = await run_in_threadpool(
            get_mcp_manager().list_tools,
            config_from_model(server),
        )
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
async def list_mounted_tools(
    operator: OperatorPrincipal = Depends(_require_tools_operator),
):
    """List the MCP tools currently mounted into the chat tool registry."""
    try:
        context = ToolContext(
            user_id=operator.user_id,
            chat_id=None,
            run_id="mcp-catalog",
        )
        definitions = await run_in_threadpool(
            get_mcp_tool_source().definitions,
            context,
        )
        return [
            McpToolInfo(name=definition.name, description=definition.description)
            for definition in definitions
        ]
    except Exception as e:
        logger.error(f"Error listing mounted MCP tools: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
