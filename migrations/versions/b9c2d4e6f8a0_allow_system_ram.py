"""Add an opt-in system RAM allowance for GPU models."""

import sqlalchemy as sa
from alembic import op


revision = "b9c2d4e6f8a0"
down_revision = "f6a9b1c3d5e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("user_settings")}
    if "llama_allow_system_ram" not in columns:
        op.add_column(
            "user_settings",
            sa.Column(
                "llama_allow_system_ram", sa.Boolean(), nullable=False, server_default=sa.false()
            ),
        )


def downgrade() -> None:
    op.drop_column("user_settings", "llama_allow_system_ram")
