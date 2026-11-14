"""Add non-secret settings for bundled email connections."""

import sqlalchemy as sa
from alembic import op


revision = "f3a5b7c9d1e4"
down_revision = "e2f4a6b8c0d2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "managed_email" not in {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("mcp_server")
    }:
        op.add_column("mcp_server", sa.Column("managed_email", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("mcp_server", "managed_email")
