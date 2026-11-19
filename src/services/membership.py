"""
src/services/membership.py
──────────────────────────
Workspace membership check shared by every service that acts inside a
workspace (documents, chats).
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import WorkspaceAccessDeniedError
from src.models.workspace import WorkspaceMember


async def assert_workspace_member(
    db: AsyncSession, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> None:
    """Raise WorkspaceAccessDeniedError unless the user belongs to the workspace."""
    result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
    )
    if not result.scalar_one_or_none():
        raise WorkspaceAccessDeniedError("You are not a member of this workspace.")
