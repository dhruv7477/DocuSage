"""question usage for the daily question limits

Revision ID: c5d8e2a71f90
Revises: a3f1b29e74cd
Create Date: 2026-11-17

Adds `question_usage`, one row per question accepted while a daily
question limit is on. The per-user and site-wide daily limits count
these rows instead of `messages`, which are deleted with their chat.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c5d8e2a71f90"
down_revision = "a3f1b29e74cd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "question_usage",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "asked_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_question_usage_asked_at", "question_usage", ["asked_at"], unique=False
    )
    op.create_index(
        "ix_question_usage_user_id_asked_at",
        "question_usage",
        ["user_id", "asked_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_question_usage_user_id_asked_at", table_name="question_usage")
    op.drop_index("ix_question_usage_asked_at", table_name="question_usage")
    op.drop_table("question_usage")
