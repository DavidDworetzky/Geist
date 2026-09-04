"""Persist resumable goal execution checkpoints.

Revision ID: e9f2a4b6c8d0
Revises: d8e1f3a5b7c9
"""

import sqlalchemy as sa
from alembic import op


revision = "e9f2a4b6c8d0"
down_revision = "d8e1f3a5b7c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_goal", sa.Column("checkpoint_json", sa.Text(), nullable=False, server_default="{}")
    )


def downgrade() -> None:
    op.drop_column("agent_goal", "checkpoint_json")
