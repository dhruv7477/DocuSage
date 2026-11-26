"""
app/schemas/chat.py
────────────────────
Pydantic schemas for Chat and Message API endpoints.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── Request Schemas ───────────────────────────────────────────────


class CreateChatRequest(BaseModel):
    workspace_id: uuid.UUID
    title: str | None = Field(default=None, max_length=500)


class ChatQueryRequest(BaseModel):
    """
    Body for POST /chat/{chat_id}/query
    Optionally filter retrieval to specific documents.
    """

    query: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] | None = None  # Scope search to these docs only


class UpdateChatTitleRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)


# ── Response Schemas ─────────────────────────────────────────────


class CitedSourceResponse(BaseModel):
    document_id: str
    source: str
    page: int
    score: float
    chunk_id: str | None = None
    heading_path: list[str] = []
    char_start: int = 0
    char_end: int = 0
    text: str = ""
    # False only on answers saved before every shown citation was checked
    # against its page text: their page is shown without a highlight.
    highlight: bool = True

    model_config = {"extra": "ignore"}


class MessageResponse(BaseModel):
    id: uuid.UUID
    chat_id: uuid.UUID
    role: str
    content: str
    sources: list[CitedSourceResponse] = []
    # "not_from_documents": the answer cites none of the documents.
    # "parts_unlinked": a claim lost every citation it had.
    label: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_message(cls, message) -> "MessageResponse":
        sources: list[CitedSourceResponse] = []
        if message.sources_json:
            try:
                # sources_json is JSONB — already a Python list, no json.loads needed
                raw = (
                    message.sources_json
                    if isinstance(message.sources_json, list)
                    else []
                )
                sources = [CitedSourceResponse(**s) for s in raw]
            except Exception:
                pass
        return cls(
            id=message.id,
            chat_id=message.chat_id,
            role=message.role.value if hasattr(message.role, "value") else message.role,
            content=message.content,
            sources=sources,
            label=message.answer_label,
            created_at=message.created_at,
        )


class ChatResponse(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    title: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatDetailResponse(ChatResponse):
    """Extended response including full message history."""

    messages: list[MessageResponse] = []


class QueryResponse(BaseModel):
    """Response for non-streaming query (POST /chat/{id}/query)."""

    answer: str
    sources: list[CitedSourceResponse] = []
    chat_id: uuid.UUID
    message_id: uuid.UUID


# ── SSE Event Schemas ─────────────────────────────────────────────


class SSETokenEvent(BaseModel):
    """Sent for each token chunk during streaming."""

    type: str = "token"
    content: str


class SSESourcesEvent(BaseModel):
    """Sent once after streaming completes — contains source citations."""

    type: str = "sources"
    sources: list[CitedSourceResponse]


class SSEDoneEvent(BaseModel):
    """Sent when the stream is complete."""

    type: str = "done"
    message_id: str


class SSEErrorEvent(BaseModel):
    """Sent if an error occurs during streaming."""

    type: str = "error"
    message: str
