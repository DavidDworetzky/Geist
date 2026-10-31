"""Align the database goal budget default with the harness."""

import sqlalchemy as sa
from alembic import op

revision = "f0a3b5c7d9e1"
down_revision = "e9f2a4b6c8d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("agent_goal") as batch:
        batch.alter_column("max_turns", existing_type=sa.Integer(), server_default="48")


def downgrade() -> None:
    with op.batch_alter_table("agent_goal") as batch:
        batch.alter_column("max_turns", existing_type=sa.Integer(), server_default="8")
