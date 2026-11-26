"""answer label on messages

Revision ID: e7b3c9d14a26
Revises: c5d8e2a71f90
Create Date: 2026-11-26

Adds `messages.answer_label`. An answer that cites none of the documents
is saved with "not_from_documents", so the chat shows it with a "Not from
your documents" label; every other message leaves it empty.
"""

import sqlalchemy as sa
from alembic import op

revision = "e7b3c9d14a26"
down_revision = "c5d8e2a71f90"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages", sa.Column("answer_label", sa.String(length=32), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("messages", "answer_label")
