"""nullable_password_hash_for_oauth

Revision ID: 50e84ad8d9e6
Revises: 49c4e57545f4
Create Date: 2026-04-20 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "50e84ad8d9e6"
down_revision: Union[str, None] = "49c4e57545f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "users", "password_hash", existing_type=sa.String(255), nullable=True
    )


def downgrade() -> None:
    op.alter_column(
        "users", "password_hash", existing_type=sa.String(255), nullable=False
    )
