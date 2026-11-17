"""messages sources_json text to jsonb

Revision ID: 831861dd2659
Revises: 50e84ad8d9e6
Create Date: 2026-04-21

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "831861dd2659"
down_revision = "50e84ad8d9e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "messages",
        "sources_json",
        existing_type=sa.Text(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using="sources_json::jsonb",
    )


def downgrade() -> None:
    op.alter_column(
        "messages",
        "sources_json",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.Text(),
        existing_nullable=True,
        postgresql_using="sources_json::text",
    )
