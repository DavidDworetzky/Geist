"""
McpServer database model: operator-configured MCP servers whose tools are
mounted into the chat tool registry when enabled.
"""

import datetime
from dataclasses import dataclass
from typing import Any

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.models.database.database import Base, SessionLocal


class McpServer(Base):
    """Connection settings for one MCP server (stdio or streamable HTTP)."""

    __tablename__ = "mcp_server"

    mcp_server_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False)

    name = Column(String, nullable=False)
    transport = Column(String, nullable=False, default="stdio")  # 'stdio' or 'http'

    # stdio transport
    command = Column(String, nullable=True)
    args = Column(JSON, default=list)
    env = Column(JSON, default=dict)
    working_directory = Column(String, nullable=True)

    # http (streamable HTTP) transport
    url = Column(String, nullable=True)
    headers = Column(JSON, default=dict)

    # Connector policy metadata. Every external server starts untrusted, and
    # email connectors stay unavailable until the security stack is installed.
    connector_kind = Column(String, nullable=False, default="custom")
    account_label = Column(String, nullable=True)
    trusted = Column(Boolean, nullable=False, default=False)
    security_required = Column(Boolean, nullable=False, default=False)
    recipient_allowlist = Column(JSON, default=list)
    max_writes_per_hour = Column(Integer, nullable=False, default=20)

    # Servers are disabled until an operator explicitly enables them; only
    # enabled servers contribute tools to chat.
    enabled = Column(Boolean, nullable=False, default=False)
    timeout_seconds = Column(Float, nullable=False, default=30.0)

    create_date = Column(DateTime, default=datetime.datetime.utcnow)
    update_date = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    user = relationship("GeistUser", backref="mcp_servers")


@dataclass
class McpServerModel:
    """Data model for a configured MCP server."""

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
    create_date: datetime.datetime
    update_date: datetime.datetime


_MUTABLE_FIELDS = (
    "name",
    "transport",
    "command",
    "args",
    "env",
    "working_directory",
    "url",
    "headers",
    "connector_kind",
    "account_label",
    "trusted",
    "security_required",
    "recipient_allowlist",
    "max_writes_per_hour",
    "enabled",
    "timeout_seconds",
)


def _to_model(server: McpServer) -> McpServerModel:
    return McpServerModel(
        mcp_server_id=server.mcp_server_id,
        user_id=server.user_id,
        name=server.name,
        transport=server.transport,
        command=server.command,
        args=list(server.args or []),
        env=dict(server.env or {}),
        working_directory=server.working_directory,
        url=server.url,
        headers=dict(server.headers or {}),
        connector_kind=server.connector_kind or "custom",
        account_label=server.account_label,
        trusted=bool(server.trusted),
        security_required=bool(server.security_required),
        recipient_allowlist=list(server.recipient_allowlist or []),
        max_writes_per_hour=int(server.max_writes_per_hour or 20),
        enabled=bool(server.enabled),
        timeout_seconds=float(server.timeout_seconds or 30.0),
        create_date=server.create_date,
        update_date=server.update_date,
    )


def list_mcp_servers(user_id: int | None = None) -> list[McpServerModel]:
    """List configured MCP servers, optionally scoped to one user."""
    with SessionLocal() as session:
        query = session.query(McpServer)
        if user_id is not None:
            query = query.filter_by(user_id=user_id)
        return [_to_model(server) for server in query.order_by(McpServer.mcp_server_id).all()]


def list_enabled_mcp_servers(user_id: int | None = None) -> list[McpServerModel]:
    """List enabled MCP servers, optionally scoped to one user."""
    with SessionLocal() as session:
        query = session.query(McpServer).filter_by(enabled=True)
        if user_id is not None:
            query = query.filter_by(user_id=user_id)
        servers = query.order_by(McpServer.mcp_server_id).all()
        if user_id is not None:
            from app.models.database.mcp_security_policy import (
                get_or_create_mcp_security_policy,
            )

            security_enabled = get_or_create_mcp_security_policy(user_id).enabled
            servers = [
                server for server in servers if not server.security_required or security_enabled
            ]
        return [_to_model(server) for server in servers]


def get_mcp_server(mcp_server_id: int) -> McpServerModel | None:
    with SessionLocal() as session:
        server = session.query(McpServer).filter_by(mcp_server_id=mcp_server_id).first()
        return _to_model(server) if server else None


def create_mcp_server(user_id: int, values: dict[str, Any]) -> McpServerModel:
    with SessionLocal() as session:
        server = McpServer(
            user_id=user_id,
            **{key: values[key] for key in _MUTABLE_FIELDS if key in values},
        )
        session.add(server)
        session.commit()
        session.refresh(server)
        return _to_model(server)


def update_mcp_server(mcp_server_id: int, updates: dict[str, Any]) -> McpServerModel | None:
    with SessionLocal() as session:
        server = session.query(McpServer).filter_by(mcp_server_id=mcp_server_id).first()
        if not server:
            return None
        for key in _MUTABLE_FIELDS:
            if key in updates:
                setattr(server, key, updates[key])
        server.update_date = datetime.datetime.utcnow()
        session.commit()
        session.refresh(server)
        return _to_model(server)


def delete_mcp_server(mcp_server_id: int) -> bool:
    with SessionLocal() as session:
        server = session.query(McpServer).filter_by(mcp_server_id=mcp_server_id).first()
        if not server:
            return False
        session.delete(server)
        session.commit()
        return True
