"""add_email_connector_policy_fields

Revision ID: c6d8e0f2a4b6
Revises: b3e5d7f9a1c3
Create Date: 2026-08-19 10:00:00.000000
"""

import sqlalchemy as sa
from alembic import op


revision = "c6d8e0f2a4b6"
down_revision = "b3e5d7f9a1c3"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("mcp_server", sa.Column("working_directory", sa.String(), nullable=True))
    op.add_column(
        "mcp_server",
        sa.Column("connector_kind", sa.String(), nullable=False, server_default="custom"),
    )
    op.add_column("mcp_server", sa.Column("account_label", sa.String(), nullable=True))
    op.add_column(
        "mcp_server",
        sa.Column("trusted", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "mcp_server",
        sa.Column("security_required", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "mcp_server",
        sa.Column("recipient_allowlist", sa.JSON(), nullable=True),
    )
    op.add_column(
        "mcp_server",
        sa.Column("max_writes_per_hour", sa.Integer(), nullable=False, server_default="20"),
    )


def downgrade():
    op.drop_column("mcp_server", "max_writes_per_hour")
    op.drop_column("mcp_server", "recipient_allowlist")
    op.drop_column("mcp_server", "security_required")
    op.drop_column("mcp_server", "trusted")
    op.drop_column("mcp_server", "account_label")
    op.drop_column("mcp_server", "connector_kind")
    op.drop_column("mcp_server", "working_directory")
