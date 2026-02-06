"""add notes table

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-02-06 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('notes',
        sa.Column('note_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False, server_default=''),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('create_date', sa.DateTime(), nullable=True),
        sa.Column('update_date', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['geist_user.user_id'], ),
        sa.PrimaryKeyConstraint('note_id')
    )

    op.create_index('idx_notes_user_id', 'notes', ['user_id'])
    op.create_index('idx_notes_update_date', 'notes', ['update_date'])
    op.create_index('idx_notes_title', 'notes', ['title'])


def downgrade() -> None:
    op.drop_index('idx_notes_title', 'notes')
    op.drop_index('idx_notes_update_date', 'notes')
    op.drop_index('idx_notes_user_id', 'notes')
    op.drop_table('notes')
