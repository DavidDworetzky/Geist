"""add provider credential table

Revision ID: a7c9e1f3b5d8
Revises: e4b7a9c2d1f0
Create Date: 2026-07-23
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a7c9e1f3b5d8'
down_revision: str | None = 'e4b7a9c2d1f0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'provider_credential',
        sa.Column('provider_credential_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('provider_id', sa.String(), nullable=False),
        sa.Column('api_key', sa.String(), nullable=False),
        sa.Column('base_url', sa.String(), nullable=True),
        sa.Column('create_date', sa.DateTime(), nullable=True),
        sa.Column('update_date', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['geist_user.user_id']),
        sa.PrimaryKeyConstraint('provider_credential_id'),
        sa.UniqueConstraint(
            'user_id', 'provider_id', name='uq_provider_credential_user_provider'
        ),
    )
    op.create_index(
        op.f('ix_provider_credential_user_id'), 'provider_credential', ['user_id'], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_provider_credential_user_id'), table_name='provider_credential')
    op.drop_table('provider_credential')
