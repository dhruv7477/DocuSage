"""
src/services/workspace_service.py
──────────────────────────────────
Workspace CRUD and member management service.

Multi-tenancy isolation is enforced here:
- Every workspace gets its own Qdrant collection and storage prefix (via model properties).
- Membership is verified before any workspace-scoped operation.
- Only the workspace owner or a workspace-level ADMIN may manage members.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import (
    InsufficientPermissionsError,
    UserNotFoundError,
    WorkspaceAccessDeniedError,
    WorkspaceNotFoundError,
)
from src.core.logger import get_logger
from src.models.user import User
from src.models.workspace import Workspace, WorkspaceMember, WorkspaceMemberRole
from src.vector_store.base import BaseVectorStore
from src.schemas.workspace import (
    CreateWorkspaceRequest,
    UpdateWorkspaceRequest,
    WorkspaceDetailResponse,
    WorkspaceMemberResponse,
    WorkspaceResponse,
)

log = get_logger(__name__)


class WorkspaceService:
    def __init__(
        self, db: AsyncSession, vector_store: BaseVectorStore | None = None
    ) -> None:
        self._db = db
        self._vector_store = vector_store

    # ── Internal helpers ──────────────────────────────────────────

    async def _get_workspace(self, workspace_id: uuid.UUID) -> Workspace:
        result = await self._db.execute(
            select(Workspace).where(Workspace.id == workspace_id)
        )
        workspace = result.scalar_one_or_none()
        if not workspace:
            raise WorkspaceNotFoundError(f"Workspace '{workspace_id}' not found.")
        return workspace

    async def _get_membership(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> WorkspaceMember | None:
        result = await self._db.execute(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def _require_member(
        self, workspace_id: uuid.UUID, user: User
    ) -> WorkspaceMember:
        membership = await self._get_membership(workspace_id, user.id)
        if not membership:
            raise WorkspaceAccessDeniedError("You are not a member of this workspace.")
        return membership

    async def _require_admin(self, workspace: Workspace, user: User) -> None:
        """Require workspace ADMIN role or workspace ownership."""
        membership = await self._get_membership(workspace.id, user.id)
        is_admin = membership and membership.role == WorkspaceMemberRole.ADMIN
        is_owner = workspace.owner_id == user.id
        if not (is_admin or is_owner):
            raise InsufficientPermissionsError("Workspace admin access required.")

    async def _member_count(self, workspace_id: uuid.UUID) -> int:
        result = await self._db.execute(
            select(func.count()).where(WorkspaceMember.workspace_id == workspace_id)
        )
        return result.scalar_one()

    # ── Workspace CRUD ────────────────────────────────────────────

    async def create_workspace(
        self, payload: CreateWorkspaceRequest, user: User
    ) -> Workspace:
        workspace = Workspace(
            name=payload.name,
            description=payload.description,
            owner_id=user.id,
        )
        self._db.add(workspace)
        await self._db.flush()

        # Creator is automatically the workspace ADMIN
        member = WorkspaceMember(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceMemberRole.ADMIN,
        )
        self._db.add(member)
        await self._db.flush()

        log.info(
            "workspace_created", workspace_id=str(workspace.id), owner=str(user.id)
        )
        return workspace

    async def list_user_workspaces(self, user: User) -> list[WorkspaceResponse]:
        # Single query: fetch workspaces + member counts in one GROUP BY
        MemberAlias = WorkspaceMember
        result = await self._db.execute(
            select(Workspace, func.count(MemberAlias.id).label("member_count"))
            .join(MemberAlias, MemberAlias.workspace_id == Workspace.id)
            .where(MemberAlias.user_id == user.id)
            .group_by(Workspace.id)
            .order_by(Workspace.created_at.desc())
        )
        rows = result.all()

        return [
            WorkspaceResponse(
                id=ws.id,
                name=ws.name,
                description=ws.description,
                owner_id=ws.owner_id,
                created_at=ws.created_at,
                updated_at=ws.updated_at,
                member_count=count,
            )
            for ws, count in rows
        ]

    async def get_workspace_detail(
        self, workspace_id: uuid.UUID, user: User
    ) -> WorkspaceDetailResponse:
        workspace = await self._get_workspace(workspace_id)
        await self._require_member(workspace_id, user)

        members_result = await self._db.execute(
            select(WorkspaceMember, User)
            .join(User, User.id == WorkspaceMember.user_id)
            .where(WorkspaceMember.workspace_id == workspace_id)
            .order_by(WorkspaceMember.created_at.asc())
        )
        member_rows = members_result.all()

        member_responses = [
            WorkspaceMemberResponse(
                id=wm.id,
                user_id=wm.user_id,
                workspace_id=wm.workspace_id,
                role=wm.role,
                user_email=u.email,
                user_full_name=u.full_name,
                joined_at=wm.created_at,
            )
            for wm, u in member_rows
        ]

        return WorkspaceDetailResponse(
            id=workspace.id,
            name=workspace.name,
            description=workspace.description,
            owner_id=workspace.owner_id,
            created_at=workspace.created_at,
            updated_at=workspace.updated_at,
            member_count=len(member_responses),
            members=member_responses,
        )

    async def update_workspace(
        self,
        workspace_id: uuid.UUID,
        payload: UpdateWorkspaceRequest,
        user: User,
    ) -> Workspace:
        workspace = await self._get_workspace(workspace_id)
        await self._require_admin(workspace, user)

        if payload.name is not None:
            workspace.name = payload.name
        if payload.description is not None:
            workspace.description = payload.description

        await self._db.flush()
        log.info("workspace_updated", workspace_id=str(workspace_id))
        return workspace

    async def delete_workspace(self, workspace_id: uuid.UUID, user: User) -> None:
        workspace = await self._get_workspace(workspace_id)
        if workspace.owner_id != user.id:
            raise InsufficientPermissionsError(
                "Only the workspace owner can delete it."
            )

        await self._db.delete(workspace)

        if self._vector_store is not None:
            try:
                await self._vector_store.delete_collection(str(workspace_id))
            except Exception as exc:
                log.warning(
                    "qdrant_collection_delete_failed",
                    workspace_id=str(workspace_id),
                    error=str(exc),
                )

        log.info("workspace_deleted", workspace_id=str(workspace_id))

    # ── Member Management ─────────────────────────────────────────

    async def add_member(
        self,
        workspace_id: uuid.UUID,
        email: str,
        role: str,
        current_user: User,
    ) -> WorkspaceMember:
        workspace = await self._get_workspace(workspace_id)
        await self._require_admin(workspace, current_user)

        target_result = await self._db.execute(select(User).where(User.email == email))
        target_user = target_result.scalar_one_or_none()
        if not target_user:
            raise UserNotFoundError(f"No user with email '{email}'.")

        existing = await self._get_membership(workspace_id, target_user.id)
        if existing:
            return existing

        try:
            member_role = WorkspaceMemberRole(role)
        except ValueError:
            member_role = WorkspaceMemberRole.MEMBER

        member = WorkspaceMember(
            workspace_id=workspace_id,
            user_id=target_user.id,
            role=member_role,
        )
        self._db.add(member)
        await self._db.flush()
        log.info(
            "member_added", workspace_id=str(workspace_id), user_id=str(target_user.id)
        )
        return member

    async def remove_member(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        current_user: User,
    ) -> None:
        workspace = await self._get_workspace(workspace_id)
        await self._require_admin(workspace, current_user)

        if user_id == workspace.owner_id:
            raise InsufficientPermissionsError("Cannot remove the workspace owner.")

        membership = await self._get_membership(workspace_id, user_id)
        if not membership:
            return  # idempotent

        await self._db.delete(membership)
        log.info("member_removed", workspace_id=str(workspace_id), user_id=str(user_id))
