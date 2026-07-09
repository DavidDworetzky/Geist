"""add agent snapshot table

Revision ID: b8c4d2e6f0a3
Revises: f1a2b3c4d5e6
Create Date: 2026-07-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8c4d2e6f0a3'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'agent_snapshot',
        sa.Column('snapshot_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('agent_identifier', sa.String(), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('reason', sa.String(), nullable=True),
        sa.Column('world_context', sa.Text(), nullable=True),
        sa.Column('task_context', sa.Text(), nullable=True),
        sa.Column('execution_context', sa.Text(), nullable=True),
        sa.Column('function_log', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('snapshot_id'),
    )
    op.create_index(
        op.f('ix_agent_snapshot_agent_identifier'),
        'agent_snapshot',
        ['agent_identifier'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_agent_snapshot_agent_identifier'), table_name='agent_snapshot')
    op.drop_table('agent_snapshot')
