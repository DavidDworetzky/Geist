"""Merge email and workspace histories, adopting pre-workspace email databases."""

import importlib

import sqlalchemy as sa
from alembic import op


revision = "b9d2e4f6a8c0"
down_revision = ("a8c1e3f5b7d9", "c6d8e0f2a4b6")
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if "workspace_key" not in {column["name"] for column in inspector.get_columns("geist_user")}:
        # The original email stack used b3e5d7f9a1c3 before main inserted
        # workspace adoption into that revision's ancestry.
        importlib.import_module(
            "migrations.versions.c6d9e2f4a7b1_add_neutral_default_workspace"
        ).upgrade()
    columns = {column["name"] for column in inspector.get_columns("mcp_server")}
    if "user_id" in columns and "workspace_id" not in columns:
        with op.batch_alter_table("mcp_server") as batch:
            batch.alter_column(
                "user_id", new_column_name="workspace_id", existing_type=sa.Integer()
            )
            batch.create_unique_constraint("uq_mcp_server_workspace_name", ["workspace_id", "name"])


def downgrade() -> None:
    pass
