"""Add MCP OAuth account metadata."""

import sqlalchemy as sa
from alembic import op


revision = "d8f0a2b4c6e8"
down_revision = "b3e5d7f9a1c3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mcp_oauth_connection",
        sa.Column("mcp_server_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("endpoint", sa.String(), nullable=False),
        sa.Column("client_id", sa.String(), nullable=False),
        sa.Column("credential_id", sa.String(32), nullable=False),
        sa.ForeignKeyConstraint(
            ["mcp_server_id"], ["mcp_server.mcp_server_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("mcp_server_id"),
        sa.UniqueConstraint("credential_id"),
    )


def downgrade() -> None:
    op.drop_table("mcp_oauth_connection")
