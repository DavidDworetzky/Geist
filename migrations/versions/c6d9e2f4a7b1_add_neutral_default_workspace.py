"""add neutral default workspace identity

Revision ID: c6d9e2f4a7b1
Revises: f5c8a1d3e7b9
Create Date: 2026-08-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "c6d9e2f4a7b1"
down_revision: str | Sequence[str] | None = "f5c8a1d3e7b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("geist_user", sa.Column("workspace_key", sa.String(), nullable=True))
    op.create_index(
        op.f("ix_geist_user_workspace_key"),
        "geist_user",
        ["workspace_key"],
        unique=True,
    )

    connection = op.get_bind()
    legacy_user_id = connection.execute(
        sa.text(
            "SELECT user_id FROM geist_user "
            "WHERE email = :email ORDER BY user_id LIMIT 1"
        ),
        {"email": "david@phantasmal.ai"},
    ).scalar()
    if legacy_user_id is not None:
        connection.execute(
            sa.text(
                "UPDATE geist_user SET "
                "workspace_key = :workspace_key, "
                "username = CASE WHEN username = :legacy_username "
                "THEN :username ELSE username END, "
                "name = CASE WHEN name = :legacy_name THEN :name ELSE name END, "
                "email = NULL, password = NULL "
                "WHERE user_id = :user_id"
            ),
            {
                "workspace_key": "default",
                "legacy_username": "ddworetzky",
                "username": "local",
                "legacy_name": "David Dworetzky",
                "name": "Local User",
                "user_id": legacy_user_id,
            },
        )
        connection.execute(
            sa.text(
                "UPDATE geist_user SET "
                "username = CASE WHEN username = :legacy_username "
                "THEN :username ELSE username END, "
                "name = CASE WHEN name = :legacy_name THEN :name ELSE name END, "
                "email = NULL, password = NULL WHERE email = :email"
            ),
            {
                "legacy_username": "ddworetzky",
                "username": "local",
                "legacy_name": "David Dworetzky",
                "name": "Local User",
                "email": "david@phantasmal.ai",
            },
        )


def downgrade() -> None:
    op.drop_index(op.f("ix_geist_user_workspace_key"), table_name="geist_user")
    op.drop_column("geist_user", "workspace_key")
