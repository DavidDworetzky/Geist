"""add agent permissions settings

Revision ID: b2e5f7a9c3d1
Revises: c7d9e1f3a5b8
Create Date: 2026-07-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "b2e5f7a9c3d1"
down_revision: str | None = "c7d9e1f3a5b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_settings",
        sa.Column("agent_permissions", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_settings", "agent_permissions")
