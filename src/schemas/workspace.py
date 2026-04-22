"""
app/schemas/workspace.py
────────────────────────
Pydantic schemas for Workspace API endpoints.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── Request Schemas ───────────────────────────────────────────────

class CreateWorkspaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=500)


class UpdateWorkspaceRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=500)


class AddMemberRequest(BaseModel):
    email: str
    role: str = "member"  # admin | member


class UpdateMemberRoleRequest(BaseModel):
    role: str  # admin | member


# ── Response Schemas ─────────────────────────────────────────────

class WorkspaceMemberResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    workspace_id: uuid.UUID
    role: str
    # Flattened user fields for convenience
    user_email: str | None = None
    user_full_name: str | None = None
    joined_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    owner_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    member_count: int = 0

    model_config = {"from_attributes": True}


class WorkspaceDetailResponse(WorkspaceResponse):
    """Extended response including member list — for workspace detail page."""
    members: list[WorkspaceMemberResponse] = []
