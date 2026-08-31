"""
McpServer database model: operator-configured MCP servers whose tools are
mounted into the chat tool registry when enabled.
"""

import datetime
from dataclasses import dataclass
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.models.database.database import Base, SessionLocal


class McpServer(Base):
    """Connection settings for one MCP server (stdio or streamable HTTP)."""

    __tablename__ = "mcp_server"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_mcp_server_workspace_name"),
    )

    mcp_server_id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("geist_user.user_id"), nullable=False)

    name = Column(String, nullable=False)
    transport = Column(String, nullable=False, default="stdio")  # 'stdio' or 'http'

    # stdio transport
    command = Column(String, nullable=True)
    args = Column(JSON, default=list)
    env = Column(JSON, default=dict)

    # http (streamable HTTP) transport
    url = Column(String, nullable=True)
    headers = Column(JSON, default=dict)

    # Servers are disabled until an operator explicitly enables them; only
    # enabled servers contribute tools to chat.
    enabled = Column(Boolean, nullable=False, default=False)
    timeout_seconds = Column(Float, nullable=False, default=30.0)

    create_date = Column(DateTime, default=datetime.datetime.utcnow)
    update_date = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    workspace = relationship("GeistUser", backref="mcp_servers")


@dataclass
class McpServerModel:
    """Data model for a configured MCP server."""

    mcp_server_id: int
    workspace_id: int
    name: str
    transport: str
    command: str | None
    args: list[str]
    env: dict[str, str]
    url: str | None
    headers: dict[str, str]
    enabled: bool
    timeout_seconds: float
    create_date: datetime.datetime
    update_date: datetime.datetime
    cwd: str | None = None
    plugin_root: str | None = None
    plugin_data_dir: str | None = None


_MUTABLE_FIELDS = (
    "name",
    "transport",
    "command",
    "args",
    "env",
    "url",
    "headers",
    "enabled",
    "timeout_seconds",
)


def _to_model(server: McpServer) -> McpServerModel:
    return McpServerModel(
        mcp_server_id=server.mcp_server_id,
        workspace_id=server.workspace_id,
        name=server.name,
        transport=server.transport,
        command=server.command,
        args=list(server.args or []),
        env=dict(server.env or {}),
        url=server.url,
        headers=dict(server.headers or {}),
        enabled=bool(server.enabled),
        timeout_seconds=float(server.timeout_seconds or 30.0),
        create_date=server.create_date,
        update_date=server.update_date,
    )


def list_mcp_servers(workspace_id: int | None = None) -> list[McpServerModel]:
    """List configured MCP servers, optionally scoped to one workspace."""
    with SessionLocal() as session:
        query = session.query(McpServer)
        if workspace_id is not None:
            query = query.filter_by(workspace_id=workspace_id)
        return [_to_model(server) for server in query.order_by(McpServer.mcp_server_id).all()]


def list_enabled_mcp_servers(workspace_id: int | None = None) -> list[McpServerModel]:
    """List enabled MCP servers, optionally scoped to one workspace."""
    with SessionLocal() as session:
        query = session.query(McpServer).filter_by(enabled=True)
        if workspace_id is not None:
            query = query.filter_by(workspace_id=workspace_id)
        servers = query.order_by(McpServer.mcp_server_id).all()
        return [_to_model(server) for server in servers]


def get_mcp_server(mcp_server_id: int) -> McpServerModel | None:
    with SessionLocal() as session:
        server = session.query(McpServer).filter_by(mcp_server_id=mcp_server_id).first()
        return _to_model(server) if server else None


def create_mcp_server(workspace_id: int, values: dict[str, Any]) -> McpServerModel:
    with SessionLocal() as session:
        server = McpServer(
            workspace_id=workspace_id,
            **{key: values[key] for key in _MUTABLE_FIELDS if key in values},
        )
        session.add(server)
        session.commit()
        session.refresh(server)
        return _to_model(server)


def update_mcp_server(
    mcp_server_id: int,
    updates: dict[str, Any],
    *,
    workspace_id: int | None = None,
) -> McpServerModel | None:
    with SessionLocal() as session:
        query = session.query(McpServer).filter_by(mcp_server_id=mcp_server_id)
        if workspace_id is not None:
            query = query.filter_by(workspace_id=workspace_id)
        server = query.first()
        if not server:
            return None
        for key in _MUTABLE_FIELDS:
            if key in updates:
                setattr(server, key, updates[key])
        server.update_date = datetime.datetime.utcnow()
        session.commit()
        session.refresh(server)
        return _to_model(server)


def delete_mcp_server(mcp_server_id: int, *, workspace_id: int | None = None) -> bool:
    with SessionLocal() as session:
        query = session.query(McpServer).filter_by(mcp_server_id=mcp_server_id)
        if workspace_id is not None:
            query = query.filter_by(workspace_id=workspace_id)
        server = query.first()
        if not server:
            return False
        session.delete(server)
        session.commit()
        return True
