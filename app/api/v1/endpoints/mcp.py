"""
API endpoints for configuring MCP (Model Context Protocol) servers.

Configured servers are persisted per user; only servers explicitly marked
enabled contribute tools to chat via the registry's MCP tool source.
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.models.tool_calling import ToolContext
from app.models.database.geist_user import get_default_user
from app.models.database.mcp_security_policy import get_or_create_mcp_security_policy
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
_REDACTED = "<redacted>"
EmailConnectorKind = Literal["gmail", "google_workspace", "outlook", "proton"]
ConnectorKind = Literal["custom", "gmail", "google_workspace", "outlook", "proton"]

_EMAIL_CONNECTOR_PROFILES = (
    {
        "kind": "gmail",
        "label": "Gmail",
        "account_scope": "Personal Google Account",
        "authentication": "OAuth 2.0 delegated authorization",
        "requirements": ["A separately installed, operator-vetted Gmail MCP server"],
    },
    {
        "kind": "google_workspace",
        "label": "Google Workspace",
        "account_scope": "Managed Google Workspace account",
        "authentication": "OAuth 2.0 under Workspace administrator policy",
        "requirements": [
            "A separately installed, operator-vetted Gmail MCP server",
            "Workspace administrator approval when organizational policy requires it",
        ],
    },
    {
        "kind": "outlook",
        "label": "Outlook / Microsoft 365",
        "account_scope": "Personal Outlook or managed Microsoft 365 account",
        "authentication": "Microsoft identity platform OAuth 2.0 delegated authorization",
        "requirements": [
            "A separately installed, operator-vetted Outlook MCP server",
            "Tenant administrator consent when organizational policy requires it",
        ],
    },
    {
        "kind": "proton",
        "label": "Proton Mail",
        "account_scope": "Paid Proton account using Proton Mail Bridge",
        "authentication": "Bridge-issued local IMAP/SMTP credentials",
        "requirements": [
            "Proton Mail Bridge installed, running, and signed in",
            "A separately installed, operator-vetted IMAP/SMTP MCP server",
        ],
    },
)


def get_current_user():
    """Get current user (placeholder - should integrate with actual auth system)."""
    return get_default_user()


def _validate_transport_fields(transport: str, command: str | None, url: str | None) -> None:
    if transport == "stdio":
        if not (command or "").strip():
            raise ValueError("stdio servers require a command")
    elif transport == "http" and (not url or not url.lower().startswith(("http://", "https://"))):
        raise ValueError("http servers require an http(s) URL")


class McpServerCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=_NAME_PATTERN)
    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = Field(default=None, max_length=1024)
    args: list[str] = Field(default_factory=list, max_length=64)
    env: dict[str, str] = Field(default_factory=dict)
    working_directory: str | None = Field(default=None, max_length=2048)
    url: str | None = Field(default=None, max_length=2048)
    headers: dict[str, str] = Field(default_factory=dict)
    connector_kind: ConnectorKind = "custom"
    account_label: str | None = Field(default=None, max_length=320)
    trusted: bool = False
    recipient_allowlist: list[str] = Field(default_factory=list, max_length=200)
    max_writes_per_hour: int = Field(default=20, ge=1, le=1000)
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
    working_directory: str | None = Field(default=None, max_length=2048)
    url: str | None = Field(default=None, max_length=2048)
    headers: dict[str, str] | None = None
    account_label: str | None = Field(default=None, max_length=320)
    trusted: bool | None = None
    recipient_allowlist: list[str] | None = Field(default=None, max_length=200)
    max_writes_per_hour: int | None = Field(default=None, ge=1, le=1000)
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
    working_directory: str | None
    url: str | None
    headers: dict[str, str]
    connector_kind: str
    account_label: str | None
    trusted: bool
    security_required: bool
    recipient_allowlist: list[str]
    max_writes_per_hour: int
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


class EmailConnectorProfile(BaseModel):
    kind: EmailConnectorKind
    label: str
    account_scope: str
    authentication: str
    requirements: list[str]


class EmailConnectorCreate(McpServerCreate):
    connector_kind: EmailConnectorKind


def _validate_email_policy(values: dict[str, Any], user_id: int) -> None:
    if (
        values.get("connector_kind") != "custom"
        and values.get("enabled")
        and not get_or_create_mcp_security_policy(user_id).enabled
    ):
        raise HTTPException(
            status_code=409,
            detail="Email connectors cannot be enabled until MCP security inspection is configured",
        )


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
        updates[field] = {
            key: stored.get(key, value) if value == _REDACTED else value
            for key, value in submitted.items()
        }


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
        values = request.model_dump()
        _validate_email_policy(values, user.user_id)
        values["security_required"] = request.connector_kind != "custom"
        server = create_mcp_server(user.user_id, values)
        _invalidate()
        return _response(server)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating MCP server: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/servers/{mcp_server_id}", response_model=McpServerResponse)
async def get_server(mcp_server_id: int):
    user = get_current_user()
    server = _owned_server_or_404(mcp_server_id, user.user_id)
    return _response(server)


@router.put("/servers/{mcp_server_id}", response_model=McpServerResponse)
async def update_server(mcp_server_id: int, request: McpServerUpdate):
    try:
        user = get_current_user()
        existing = _owned_server_or_404(mcp_server_id, user.user_id)
        updates = request.model_dump(exclude_unset=True)
        _preserve_redacted_secrets(updates, existing)
        _validate_email_policy(
            {
                "connector_kind": existing.connector_kind,
                "enabled": updates.get("enabled", existing.enabled),
            },
            user.user_id,
        )

        merged_transport = updates.get("transport", existing.transport)
        merged_command = updates.get("command", existing.command)
        merged_url = updates.get("url", existing.url)
        try:
            _validate_transport_fields(merged_transport, merged_command, merged_url)
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


@router.get("/email-connectors/profiles", response_model=list[EmailConnectorProfile])
async def list_email_connector_profiles():
    """Return supported account profiles without selecting a third-party server."""
    return [EmailConnectorProfile.model_validate(profile) for profile in _EMAIL_CONNECTOR_PROFILES]


@router.post("/email-connectors", response_model=McpServerResponse, status_code=201)
async def create_email_connector(request: EmailConnectorCreate):
    """Create a disabled, untrusted email connector owned by the default user."""
    values = request.model_dump()
    values["enabled"] = False
    values["trusted"] = False
    values["security_required"] = True
    user = get_current_user()
    if any(server.name == request.name for server in list_mcp_servers(user.user_id)):
        raise HTTPException(
            status_code=409, detail=f"An MCP server named '{request.name}' already exists"
        )
    server = create_mcp_server(user.user_id, values)
    _invalidate()
    return _response(server)


@router.delete("/servers/{mcp_server_id}", status_code=204)
async def delete_server(mcp_server_id: int):
    try:
        user = get_current_user()
        _owned_server_or_404(mcp_server_id, user.user_id)
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
    user = get_current_user()
    server = _owned_server_or_404(mcp_server_id, user.user_id)
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
        user = get_current_user()
        context = ToolContext(user_id=user.user_id, chat_id=None, run_id="mcp-catalog")
        return [
            McpToolInfo(name=definition.name, description=definition.description)
            for definition in get_mcp_tool_source().definitions(context)
        ]
    except Exception as e:
        logger.error(f"Error listing mounted MCP tools: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
