"""add prompt schedules and run provenance

Revision ID: a4d9c7e2f6b1
Revises: b3e5d7f9a1c3
Create Date: 2026-09-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "a4d9c7e2f6b1"
down_revision: str | None = "b3e5d7f9a1c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Legacy unversioned databases are classified from current SQLAlchemy
    # metadata before being stamped at an older revision. In that adoption
    # path these tables may already exist even though Alembic still needs to
    # advance through this migration.
    existing_tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "prompt_schedule" not in existing_tables:
        op.create_table(
            "prompt_schedule",
            sa.Column("prompt_schedule_id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("prompt", sa.Text(), nullable=False),
            sa.Column("cron_expression", sa.String(), nullable=False),
            sa.Column("timezone", sa.String(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("inference_config", sa.JSON(), nullable=False),
            sa.Column("next_run_at", sa.DateTime(), nullable=True),
            sa.Column("last_enqueued_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["geist_user.user_id"]),
            sa.PrimaryKeyConstraint("prompt_schedule_id"),
        )
        op.create_index(
            op.f("ix_prompt_schedule_user_id"),
            "prompt_schedule",
            ["user_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_prompt_schedule_next_run_at"),
            "prompt_schedule",
            ["next_run_at"],
            unique=False,
        )

    if "prompt_schedule_run" not in existing_tables:
        op.create_table(
            "prompt_schedule_run",
            sa.Column(
                "prompt_schedule_run_id", sa.Integer(), autoincrement=True, nullable=False
            ),
            sa.Column("prompt_schedule_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("job_id", sa.Integer(), nullable=False),
            sa.Column("trigger_type", sa.String(), nullable=False),
            sa.Column("scheduled_for", sa.DateTime(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["job_id"], ["job.job_id"]),
            sa.ForeignKeyConstraint(["user_id"], ["geist_user.user_id"]),
            sa.PrimaryKeyConstraint("prompt_schedule_run_id"),
            sa.UniqueConstraint(
                "prompt_schedule_id",
                "scheduled_for",
                name="uq_prompt_schedule_run_occurrence",
            ),
        )
        op.create_index(
            op.f("ix_prompt_schedule_run_prompt_schedule_id"),
            "prompt_schedule_run",
            ["prompt_schedule_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_prompt_schedule_run_user_id"),
            "prompt_schedule_run",
            ["user_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_prompt_schedule_run_job_id"),
            "prompt_schedule_run",
            ["job_id"],
            unique=True,
        )


def downgrade() -> None:
    op.drop_index(op.f("ix_prompt_schedule_run_job_id"), table_name="prompt_schedule_run")
    op.drop_index(op.f("ix_prompt_schedule_run_user_id"), table_name="prompt_schedule_run")
    op.drop_index(
        op.f("ix_prompt_schedule_run_prompt_schedule_id"), table_name="prompt_schedule_run"
    )
    op.drop_table("prompt_schedule_run")
    op.drop_index(op.f("ix_prompt_schedule_next_run_at"), table_name="prompt_schedule")
    op.drop_index(op.f("ix_prompt_schedule_user_id"), table_name="prompt_schedule")
    op.drop_table("prompt_schedule")
