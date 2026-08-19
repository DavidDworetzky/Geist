"""add llama compute settings

Revision ID: c7d9e1f3a5b8
Revises: f5c8a1d3e7b9
Create Date: 2026-08-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "c7d9e1f3a5b8"
down_revision: str | None = "f5c8a1d3e7b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("user_settings", sa.Column("llama_backend", sa.String(), nullable=True))
    op.add_column(
        "user_settings",
        sa.Column("llama_gpu_device_ids", sa.JSON(), nullable=True),
    )
    op.execute(
        "UPDATE user_settings SET llama_gpu_device_ids = '[]' WHERE llama_gpu_device_ids IS NULL"
    )


def downgrade() -> None:
    op.drop_column("user_settings", "llama_gpu_device_ids")
    op.drop_column("user_settings", "llama_backend")
