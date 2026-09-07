"""add_mcp_server_table

Revision ID: b3e5d7f9a1c3
Revises: f5c8a1d3e7b9
Create Date: 2026-07-28 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "b3e5d7f9a1c3"
down_revision = "c6d9e2f4a7b1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "mcp_server",
        sa.Column("mcp_server_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("transport", sa.String(), nullable=False),
        sa.Column("command", sa.String(), nullable=True),
        sa.Column("args", sa.JSON(), nullable=True),
        sa.Column("env", sa.JSON(), nullable=True),
        sa.Column("url", sa.String(), nullable=True),
        sa.Column("headers", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("timeout_seconds", sa.Float(), nullable=False),
        sa.Column("create_date", sa.DateTime(), nullable=True),
        sa.Column("update_date", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["geist_user.user_id"],
        ),
        sa.PrimaryKeyConstraint("mcp_server_id"),
        sa.UniqueConstraint("workspace_id", "name", name="uq_mcp_server_workspace_name"),
    )


def downgrade():
    op.drop_table("mcp_server")
