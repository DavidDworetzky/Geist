"""merge native runtime and hierarchical memory heads

Revision ID: f5c8a1d3e7b9
Revises: a4d6e8f0b2c4, e4b7a9c2d1f0
Create Date: 2026-07-24
"""

from collections.abc import Sequence


revision: str = "f5c8a1d3e7b9"
down_revision: str | Sequence[str] | None = (
    "a4d6e8f0b2c4",
    "e4b7a9c2d1f0",
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
