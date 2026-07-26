"""add local artifact selection

Revision ID: a4d6e8f0b2c4
Revises: c3a1f5e7d9b2
Create Date: 2026-07-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "a4d6e8f0b2c4"
down_revision: str | None = "c3a1f5e7d9b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_settings",
        sa.Column("default_local_artifact_id", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_settings", "default_local_artifact_id")
