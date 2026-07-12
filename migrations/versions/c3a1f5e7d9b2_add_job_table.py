"""add job table

Revision ID: c3a1f5e7d9b2
Revises: b8c4d2e6f0a3
Create Date: 2026-07-11
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c3a1f5e7d9b2'
down_revision: str | None = 'b8c4d2e6f0a3'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'job',
        sa.Column('job_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('kind', sa.String(), nullable=False),
        sa.Column('payload', sa.Text(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('max_attempts', sa.Integer(), nullable=False),
        sa.Column('run_after', sa.DateTime(), nullable=False),
        sa.Column('result', sa.Text(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('job_id'),
    )
    op.create_index(op.f('ix_job_kind'), 'job', ['kind'], unique=False)
    op.create_index(op.f('ix_job_status'), 'job', ['status'], unique=False)
    op.create_index(op.f('ix_job_run_after'), 'job', ['run_after'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_job_run_after'), table_name='job')
    op.drop_index(op.f('ix_job_status'), table_name='job')
    op.drop_index(op.f('ix_job_kind'), table_name='job')
    op.drop_table('job')
