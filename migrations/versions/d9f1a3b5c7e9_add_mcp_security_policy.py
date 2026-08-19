"""add_mcp_security_policy

Revision ID: d9f1a3b5c7e9
Revises: c6d8e0f2a4b6
Create Date: 2026-08-19 11:00:00.000000
"""

import sqlalchemy as sa
from alembic import op


revision = "d9f1a3b5c7e9"
down_revision = "c6d8e0f2a4b6"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "mcp_security_policy",
        sa.Column("mcp_security_policy_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("inspect_tool_metadata", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "inspect_outbound_arguments", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
        sa.Column(
            "inspect_inbound_results", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
        sa.Column("deterministic_scanner", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("model_mode", sa.String(), nullable=False, server_default="mirror"),
        sa.Column("create_date", sa.DateTime(), nullable=True),
        sa.Column("update_date", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["geist_user.user_id"]),
        sa.PrimaryKeyConstraint("mcp_security_policy_id"),
        sa.UniqueConstraint("user_id"),
    )


def downgrade():
    op.drop_table("mcp_security_policy")
