"""OAuth metadata only; account credentials live in the private runtime store."""

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.models.database.database import Base


class McpOAuthConnection(Base):
    __tablename__ = "mcp_oauth_connection"

    mcp_server_id = Column(
        Integer, ForeignKey("mcp_server.mcp_server_id", ondelete="CASCADE"), primary_key=True
    )
    provider: str = Column(String, nullable=False)
    endpoint: str = Column(String, nullable=False)
    client_id: str = Column(String, nullable=False)
    credential_id: str = Column(String(32), nullable=False, unique=True)

    server = relationship("McpServer", back_populates="oauth_connection")
