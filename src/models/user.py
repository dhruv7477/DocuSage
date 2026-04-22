"""
src/models/user.py
──────────────────
User model — represents a registered account.

Role hierarchy:
    SUPER_ADMIN → platform owner (can see all workspaces)
    ADMIN       → workspace admin (manages members in their workspace)
    MEMBER      → regular user (can upload docs, query within workspace)
"""

import uuid
from enum import Enum

from sqlalchemy import Boolean, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MEMBER = "member"


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Core user entity.

    Notes:
    - email is the unique identity — normalized to lowercase on insert
    - password_hash stores bcrypt hash; never store plaintext
    - is_active: soft-disable accounts without deletion
    - is_email_verified: gate features behind email confirmation
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        String(50), nullable=False, default=UserRole.MEMBER
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_email_verified: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationships ────────────────────────────────────────────
    owned_workspaces: Mapped[list["Workspace"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Workspace", back_populates="owner", foreign_keys="Workspace.owner_id"
    )
    workspace_memberships: Mapped[list["WorkspaceMember"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "WorkspaceMember", back_populates="user"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"
