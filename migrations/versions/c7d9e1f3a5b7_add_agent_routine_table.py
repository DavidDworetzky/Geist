"""add agent routine table

Revision ID: c7d9e1f3a5b7
Revises: b2e5f7a9c3d1
Create Date: 2026-08-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "c7d9e1f3a5b7"
down_revision: str | None = "b2e5f7a9c3d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_routine",
        sa.Column("routine_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("geist_user.user_id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("prompt", sa.String(), nullable=False),
        sa.Column("interval_minutes", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column("next_run_at", sa.DateTime(), nullable=True),
        sa.Column("create_date", sa.DateTime(), nullable=True),
        sa.Column("update_date", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_agent_routine_next_run", "agent_routine", ["enabled", "next_run_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_agent_routine_next_run", table_name="agent_routine")
    op.drop_table("agent_routine")
