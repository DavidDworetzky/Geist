"""Persist routine execution outcomes without changing existing schedules."""

import sqlalchemy as sa
from alembic import op


revision = "g7b0c2d4e6f8"
down_revision = "e5f8a0b2c4d6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("agent_routine")}
    for name, length in (("last_status", 32), ("last_error", 500)):
        if name not in columns:
            op.add_column("agent_routine", sa.Column(name, sa.String(length), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_routine", "last_error")
    op.drop_column("agent_routine", "last_status")
