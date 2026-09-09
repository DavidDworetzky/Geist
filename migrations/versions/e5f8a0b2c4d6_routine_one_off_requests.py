"""Track an explicit one-off request without changing recurrence or existing data."""

import sqlalchemy as sa
from alembic import op

revision = "e5f8a0b2c4d6"
down_revision = "d4e7f9a1b3c5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Metadata-created legacy databases may already contain this additive field.
    if "run_once_requested" not in {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("agent_routine")
    }:
        op.add_column(
            "agent_routine",
            sa.Column(
                "run_once_requested",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )
    if "ix_agent_routine_next_run" not in {
        index["name"] for index in sa.inspect(op.get_bind()).get_indexes("agent_routine")
    }:
        op.create_index("ix_agent_routine_next_run", "agent_routine", ["enabled", "next_run_at"])


def downgrade() -> None:
    op.drop_column("agent_routine", "run_once_requested")
