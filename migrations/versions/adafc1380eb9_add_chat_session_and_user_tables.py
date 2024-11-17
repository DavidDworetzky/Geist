"""add chat session and user tables.

Revision ID: adafc1380eb9
Revises: d7f88cab37e0
Create Date: 2024-11-17 05:44:54.529385

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'adafc1380eb9'
down_revision: Union[str, None] = 'd7f88cab37e0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=True),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('password', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('user_id')
    )
    
    op.create_table('chat_session',
        sa.Column('chat_session_id', sa.Integer(), nullable=False),
        sa.Column('chat_history', sa.String(), nullable=True),
        sa.Column('create_date', sa.DateTime(), nullable=True),
        sa.Column('update_date', sa.DateTime(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.user_id'], ),
        sa.PrimaryKeyConstraint('chat_session_id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('chat_session')
    op.drop_table('user')
    # ### end Alembic commands ###
