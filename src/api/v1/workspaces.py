"""
src/api/v1/workspaces.py
────────────────────────
Workspace CRUD and member management endpoints.
All routes require authentication via Depends(get_current_user).
"""

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.dependencies import get_current_user
from src.models.user import User
from src.schemas.workspace import (
    AddMemberRequest,
    CreateWorkspaceRequest,
    UpdateWorkspaceRequest,
    WorkspaceDetailResponse,
    WorkspaceMemberResponse,
    WorkspaceResponse,
)
from src.services.workspace_service import WorkspaceService
from src.vector_store.factory import get_vector_store

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


def _get_service(db: AsyncSession = Depends(get_db)) -> WorkspaceService:
    return WorkspaceService(db, vector_store=get_vector_store())


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workspace",
)
async def create_workspace(
    payload: CreateWorkspaceRequest,
    current_user: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_get_service),
) -> WorkspaceResponse:
    workspace = await service.create_workspace(payload, current_user)
    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        description=workspace.description,
        owner_id=workspace.owner_id,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
        member_count=1,
    )


@router.get(
    "",
    response_model=list[WorkspaceResponse],
    summary="List all workspaces current user is a member of",
)
async def list_workspaces(
    current_user: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_get_service),
) -> list[WorkspaceResponse]:
    return await service.list_user_workspaces(current_user)


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceDetailResponse,
    summary="Get workspace detail with member list",
)
async def get_workspace(
    workspace_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_get_service),
) -> WorkspaceDetailResponse:
    return await service.get_workspace_detail(workspace_id, current_user)


@router.patch(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    summary="Update workspace name or description (admin only)",
)
async def update_workspace(
    workspace_id: uuid.UUID,
    payload: UpdateWorkspaceRequest,
    current_user: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_get_service),
) -> WorkspaceResponse:
    workspace = await service.update_workspace(workspace_id, payload, current_user)
    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        description=workspace.description,
        owner_id=workspace.owner_id,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
    )


@router.delete(
    "/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete workspace (owner only)",
)
async def delete_workspace(
    workspace_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_get_service),
) -> None:
    await service.delete_workspace(workspace_id, current_user)


# ── Member Management ─────────────────────────────────────────────

@router.post(
    "/{workspace_id}/members",
    response_model=WorkspaceMemberResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a member to the workspace by email (admin only)",
)
async def add_member(
    workspace_id: uuid.UUID,
    payload: AddMemberRequest,
    current_user: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_get_service),
) -> WorkspaceMemberResponse:
    member = await service.add_member(
        workspace_id, payload.email, payload.role, current_user
    )
    return WorkspaceMemberResponse(
        id=member.id,
        user_id=member.user_id,
        workspace_id=member.workspace_id,
        role=member.role,
        joined_at=member.created_at,
    )


@router.delete(
    "/{workspace_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a member from the workspace (admin only)",
)
async def remove_member(
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: WorkspaceService = Depends(_get_service),
) -> None:
    await service.remove_member(workspace_id, user_id, current_user)