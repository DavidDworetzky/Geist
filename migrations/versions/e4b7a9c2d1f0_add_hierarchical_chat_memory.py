"""add hierarchical chat memory

Revision ID: e4b7a9c2d1f0
Revises: c3a1f5e7d9b2
Create Date: 2026-07-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "e4b7a9c2d1f0"
down_revision: str | None = "c3a1f5e7d9b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("job") as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("dedupe_key", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("locked_at", sa.DateTime(), nullable=True))
        batch_op.create_foreign_key(
            "fk_job_user_id",
            "geist_user",
            ["user_id"],
            ["user_id"],
        )
        batch_op.create_index(op.f("ix_job_user_id"), ["user_id"], unique=False)
        batch_op.create_index(op.f("ix_job_dedupe_key"), ["dedupe_key"], unique=False)

    op.create_table(
        "memory_folder",
        sa.Column("folder_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("color", sa.String(length=20), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["geist_user.user_id"]),
        sa.PrimaryKeyConstraint("folder_id"),
        sa.UniqueConstraint("user_id", "name", name="uq_memory_folder_user_name"),
    )
    op.create_index(op.f("ix_memory_folder_user_id"), "memory_folder", ["user_id"], unique=False)

    with op.batch_alter_table("chat_session") as batch_op:
        batch_op.add_column(
            sa.Column("memory_enabled", sa.Boolean(), nullable=False, server_default=sa.true())
        )
        batch_op.add_column(
            sa.Column(
                "memory_mode",
                sa.String(length=20),
                nullable=False,
                server_default="public",
            )
        )
        batch_op.add_column(sa.Column("folder_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("memory_revision", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column(
                "memory_processed_revision",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(sa.Column("memory_last_activity_at", sa.DateTime(), nullable=True))
        batch_op.create_foreign_key(
            "fk_chat_session_folder_id",
            "memory_folder",
            ["folder_id"],
            ["folder_id"],
        )
        batch_op.create_index(op.f("ix_chat_session_folder_id"), ["folder_id"], unique=False)

    op.create_table(
        "memory_record",
        sa.Column("memory_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("scope", sa.String(length=20), nullable=False),
        sa.Column("record_type", sa.String(length=40), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("folder_id", sa.Integer(), nullable=True),
        sa.Column("chat_session_id", sa.Integer(), nullable=True),
        sa.Column("source_chat_session_id", sa.Integer(), nullable=True),
        sa.Column("source_from_revision", sa.Integer(), nullable=True),
        sa.Column("source_through_revision", sa.Integer(), nullable=True),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chat_session_id"],
            ["chat_session.chat_session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["folder_id"],
            ["memory_folder.folder_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_chat_session_id"],
            ["chat_session.chat_session_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["geist_user.user_id"]),
        sa.PrimaryKeyConstraint("memory_id"),
        sa.UniqueConstraint(
            "chat_session_id",
            "source_through_revision",
            "record_type",
            "content_hash",
            name="uq_memory_record_chat_revision_type",
        ),
    )
    for column in (
        "active",
        "chat_session_id",
        "content_hash",
        "folder_id",
        "record_type",
        "scope",
        "source_chat_session_id",
        "user_id",
    ):
        op.create_index(
            op.f(f"ix_memory_record_{column}"),
            "memory_record",
            [column],
            unique=False,
        )

    op.create_table(
        "memory_embedding",
        sa.Column("memory_id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memory_record.memory_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("memory_id"),
    )


def downgrade() -> None:
    op.drop_table("memory_embedding")
    for column in (
        "user_id",
        "source_chat_session_id",
        "scope",
        "record_type",
        "folder_id",
        "content_hash",
        "chat_session_id",
        "active",
    ):
        op.drop_index(op.f(f"ix_memory_record_{column}"), table_name="memory_record")
    op.drop_table("memory_record")
    with op.batch_alter_table("chat_session") as batch_op:
        batch_op.drop_index(op.f("ix_chat_session_folder_id"))
        batch_op.drop_constraint("fk_chat_session_folder_id", type_="foreignkey")
        for column in (
            "memory_last_activity_at",
            "memory_processed_revision",
            "memory_revision",
            "folder_id",
            "memory_mode",
            "memory_enabled",
        ):
            batch_op.drop_column(column)
    op.drop_index(op.f("ix_memory_folder_user_id"), table_name="memory_folder")
    op.drop_table("memory_folder")
    with op.batch_alter_table("job") as batch_op:
        batch_op.drop_index(op.f("ix_job_dedupe_key"))
        batch_op.drop_index(op.f("ix_job_user_id"))
        batch_op.drop_constraint("fk_job_user_id", type_="foreignkey")
        batch_op.drop_column("locked_at")
        batch_op.drop_column("dedupe_key")
        batch_op.drop_column("user_id")
