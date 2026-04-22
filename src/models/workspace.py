"""
src/models/workspace.py
───────────────────────
Workspace = one tenant in the multi-tenant system.

Every workspace has:
- Its own isolated vector namespace in Qdrant
- Its own isolated S3/MinIO prefix
- Its own isolated document records and chat history
- Its own member list with per-member roles
"""

import uuid
from enum import Enum

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class WorkspaceMemberRole(str, Enum):
    """
    Roles within a workspace (separate from global UserRole).
    A global MEMBER can be a workspace-level ADMIN for their own workspace.
    """
    ADMIN = "admin"    # Can manage members, delete workspace, upload docs
    MEMBER = "member"  # Can upload docs and query


class Workspace(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Tenant isolation unit.

    Qdrant collection name: f"{settings.qdrant_collection_prefix}_ws_{id}"
    Storage prefix: f"workspaces/{id}/"
    """

    __tablename__ = "workspaces"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Derived property used everywhere for Qdrant collection naming
    @property
    def vector_collection_name(self) -> str:
        from src.configs.config import get_settings
        prefix = get_settings().qdrant_collection_prefix
        return f"{prefix}_ws_{self.id}"

    @property
    def storage_prefix(self) -> str:
        return f"workspaces/{self.id}"

    # ── Relationships ────────────────────────────────────────────
    owner: Mapped["User"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "User", back_populates="owned_workspaces", foreign_keys=[owner_id]
    )
    members: Mapped[list["WorkspaceMember"]] = relationship(
        "WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Document", back_populates="workspace", cascade="all, delete-orphan"
    )
    chats: Mapped[list["Chat"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Chat", back_populates="workspace", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Workspace id={self.id} name={self.name}>"


class WorkspaceMember(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """
    Junction table: User ↔ Workspace with role.

    Unique constraint ensures a user can only have one role per workspace.
    """

    __tablename__ = "workspace_members"
    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[WorkspaceMemberRole] = mapped_column(
        String(50), nullable=False, default=WorkspaceMemberRole.MEMBER
    )

    # ── Relationships ────────────────────────────────────────────
    workspace: Mapped["Workspace"] = relationship(
        "Workspace", back_populates="members"
    )
    user: Mapped["User"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "User", back_populates="workspace_memberships"
    )

    def __repr__(self) -> str:
        return (
            f"<WorkspaceMember workspace={self.workspace_id} "
            f"user={self.user_id} role={self.role}>"
        )
