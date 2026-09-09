"""add agent goal table

Revision ID: d8e1f3a5b7c9
Revises: c7d9e1f3a5b7
Create Date: 2026-09-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "d8e1f3a5b7c9"
down_revision: str | None = "c7d9e1f3a5b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_settings",
        sa.Column(
            "agentic_mode_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    op.create_table(
        "agent_goal",
        sa.Column("goal_id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("geist_user.user_id"), nullable=False),
        sa.Column(
            "chat_id",
            sa.Integer(),
            sa.ForeignKey("chat_session.chat_session_id"),
            nullable=True,
        ),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("plan_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("turns_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_turns", sa.Integer(), nullable=False, server_default="8"),
        sa.Column("completion_summary", sa.Text(), nullable=True),
        sa.Column("completion_evidence_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("create_date", sa.DateTime(), nullable=True),
        sa.Column("update_date", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_agent_goal_user_id", "agent_goal", ["user_id"])
    op.create_index("ix_agent_goal_chat_id", "agent_goal", ["chat_id"])
    op.create_index("ix_agent_goal_run_id", "agent_goal", ["run_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_agent_goal_run_id", table_name="agent_goal")
    op.drop_index("ix_agent_goal_chat_id", table_name="agent_goal")
    op.drop_index("ix_agent_goal_user_id", table_name="agent_goal")
    op.drop_table("agent_goal")
    op.drop_column("user_settings", "agentic_mode_enabled")
