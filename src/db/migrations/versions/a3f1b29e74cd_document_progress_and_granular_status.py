"""document progress and granular ingestion status

Revision ID: a3f1b29e74cd
Revises: 831861dd2659
Create Date: 2026-05-01

Adds a `progress` column on `documents` (0–100, default 0) so the
frontend can render an indexing progress bar.

The granular ingestion sub-states (parsing/chunking/embedding/indexing)
are stored in the existing `status` column — no enum migration is needed
because `status` is a VARCHAR, not a Postgres enum type.
"""
from alembic import op
import sqlalchemy as sa

revision = 'a3f1b29e74cd'
down_revision = '831861dd2659'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
    )
    # Drop the server_default after backfilling; new rows are populated by
    # the application so we don't need a DB-level default going forward.
    op.alter_column('documents', 'progress', server_default=None)


def downgrade() -> None:
    op.drop_column('documents', 'progress')
