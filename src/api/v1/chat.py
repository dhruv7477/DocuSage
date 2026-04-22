"""
src/api/v1/chat.py
───────────────────
Chat endpoints — the core product experience.

Endpoints:
    POST   /chat                        Create a new chat session
    GET    /chat                        List chats in a workspace
    GET    /chat/{chat_id}              Get chat with message history
    PATCH  /chat/{chat_id}/title        Rename chat
    DELETE /chat/{chat_id}              Delete chat + all messages
    POST   /chat/{chat_id}/query        Non-streaming RAG query
    GET    /chat/{chat_id}/stream       Streaming SSE RAG query
    GET    /chat/{chat_id}/messages     Get message history

SSE Streaming Protocol:
    The client connects via GET /chat/{id}/stream?query=...
    Events are sent as:
        data: {"type":"token","content":"Hello"}
        data: {"type":"token","content":" world"}
        data: {"type":"sources","sources":[...]}
        data: {"type":"done","message_id":"..."}

    On error:
        data: {"type":"error","message":"..."}

Why GET for streaming (not POST):
    EventSource API in browsers only supports GET requests.
    The query is passed as a URL parameter.
    For production, use a POST-based SSE library or WebSockets.
"""

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.dependencies import get_current_user
from src.models.user import User
from src.providers.embedding.factory import get_embedding_provider
from src.providers.llm.factory import get_llm_provider
from src.schemas.chat import (
    ChatDetailResponse,
    ChatQueryRequest,
    ChatResponse,
    CreateChatRequest,
    MessageResponse,
    QueryResponse,
    UpdateChatTitleRequest,
)
from src.services.chat_service import ChatService
from src.services.rag_service import RAGService
from src.vector_store.factory import get_vector_store

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Dependency factories ──────────────────────────────────────────

def _get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    return ChatService(db)


def _get_rag_service() -> RAGService:
    from src.configs.config import get_settings
    from src.retrieval.pipeline import RetrievalPipeline
    from src.retrieval.query_rewriter import QueryRewriter, RewriteStrategy
    from src.retrieval.reranker import get_reranker

    _settings = get_settings()
    llm = get_llm_provider()
    embedder = get_embedding_provider()
    vector_store = get_vector_store()

    pipeline = RetrievalPipeline(
        embedder=embedder,
        vector_store=vector_store,
        rewriter=QueryRewriter(llm=llm),
        reranker=get_reranker(
            strategy=_settings.reranker,
            llm=llm,
            model_name=_settings.reranker_model,
        ),
        rewrite_strategy=RewriteStrategy(_settings.rewriter_strategy),
        candidate_k=_settings.retrieval_candidate_k,
        final_top_k=_settings.rag_top_k,
        score_threshold=_settings.rag_score_threshold,
        use_bm25=_settings.enable_bm25,
    )

    return RAGService(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        retrieval_pipeline=pipeline,
    )


# ── Session CRUD ─────────────────────────────────────────────────

@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
async def create_chat(
    payload: CreateChatRequest,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(_get_chat_service),
) -> ChatResponse:
    chat = await chat_service.create_chat(
        workspace_id=payload.workspace_id,
        user=current_user,
        title=payload.title,
    )
    return ChatResponse.model_validate(chat)


@router.get(
    "",
    response_model=list[ChatResponse],
    summary="List all chats in a workspace",
)
async def list_chats(
    workspace_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(_get_chat_service),
) -> list[ChatResponse]:
    chats = await chat_service.list_chats(workspace_id, current_user, limit=limit, offset=offset)
    return [ChatResponse.model_validate(c) for c in chats]


@router.get(
    "/{chat_id}",
    response_model=ChatDetailResponse,
    summary="Get chat detail with full message history",
)
async def get_chat(
    chat_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(_get_chat_service),
) -> ChatDetailResponse:
    chat = await chat_service.get_chat(chat_id, current_user)
    messages = await chat_service.get_messages(chat_id, current_user)
    return ChatDetailResponse(
        **ChatResponse.model_validate(chat).model_dump(),
        messages=[MessageResponse.from_message(m) for m in messages],
    )


@router.patch(
    "/{chat_id}/title",
    response_model=ChatResponse,
    summary="Rename a chat session",
)
async def update_chat_title(
    chat_id: uuid.UUID,
    payload: UpdateChatTitleRequest,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(_get_chat_service),
) -> ChatResponse:
    chat = await chat_service.update_chat_title(chat_id, payload.title, current_user)
    return ChatResponse.model_validate(chat)


@router.delete(
    "/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session and all its messages",
)
async def delete_chat(
    chat_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(_get_chat_service),
) -> None:
    await chat_service.delete_chat(chat_id, current_user)


# ── Messages ─────────────────────────────────────────────────────

@router.get(
    "/{chat_id}/messages",
    response_model=list[MessageResponse],
    summary="Get paginated message history for a chat",
)
async def get_messages(
    chat_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(_get_chat_service),
) -> list[MessageResponse]:
    messages = await chat_service.get_messages(chat_id, current_user, limit=limit)
    return [MessageResponse.from_message(m) for m in messages]


# ── Non-Streaming Query ───────────────────────────────────────────

@router.post(
    "/{chat_id}/query",
    response_model=QueryResponse,
    summary="Send a query and get a complete answer (non-streaming)",
)
async def query(
    chat_id: uuid.UUID,
    payload: ChatQueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(_get_rag_service),
) -> QueryResponse:
    """
    Full RAG pipeline in a single request-response cycle.
    Use this for programmatic access or when SSE is not available.
    """
    chat_service = ChatService(db)

    # Verify ownership + load chat
    chat = await chat_service.get_chat(chat_id, current_user)

    # Load history for context
    history = await chat_service.load_history(chat_id)

    # Persist user message
    await chat_service.save_user_message(chat_id, payload.query)

    # Run RAG pipeline
    rag_response = await rag_service.generate(
        query=payload.query,
        workspace_id=chat.workspace_id,
        user=current_user,
        history_messages=history,
        document_ids=payload.document_ids,
    )

    # Persist assistant message with sources
    assistant_msg = await chat_service.save_assistant_message(
        chat_id=chat_id,
        content=rag_response.answer,
        sources=rag_response.sources,
    )

    # Auto-title on first message
    if chat.message_count <= 2:
        title = ChatService.generate_title(payload.query)
        await chat_service.update_chat_title(chat_id, title, current_user)

    from src.schemas.chat import CitedSourceResponse
    return QueryResponse(
        answer=rag_response.answer,
        sources=[
            CitedSourceResponse(
                document_id=s.document_id,
                source=s.source_filename,
                page=s.page_number,
                score=s.relevance_score,
            )
            for s in rag_response.sources
        ],
        chat_id=chat_id,
        message_id=assistant_msg.id,
    )


# ── SSE Streaming Query ───────────────────────────────────────────

@router.get(
    "/{chat_id}/stream",
    summary="Stream a RAG answer via Server-Sent Events (SSE)",
    response_class=StreamingResponse,
)
async def stream_query(
    chat_id: uuid.UUID,
    query: str = Query(..., min_length=1, max_length=4000),
    document_ids: str | None = Query(default=None, description="Comma-separated document IDs to scope search"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(_get_rag_service),
) -> StreamingResponse:
    """
    SSE streaming endpoint for real-time chat UI.

    Protocol:
        data: {"type":"token","content":"<text>"}  (repeated)
        data: {"type":"sources","sources":[...]}   (once, after stream)
        data: {"type":"done","message_id":"<id>"}  (final event)

    Connect using fetchEventSource with Authorization header:
        fetchEventSource(`/api/v1/chat/${chatId}/stream?query=...`, {
            headers: { Authorization: `Bearer ${token}` }
        })
    """
    doc_id_list = [d.strip() for d in document_ids.split(",")] if document_ids else None

    return StreamingResponse(
        _sse_generator(
            chat_id=chat_id,
            query=query,
            document_ids=doc_id_list,
            current_user=current_user,
            db=db,
            rag_service=rag_service,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )


async def _sse_generator(
    chat_id: uuid.UUID,
    query: str,
    document_ids: list[str] | None,
    current_user: User,
    db: AsyncSession,
    rag_service: RAGService,
) -> AsyncIterator[str]:
    """
    Core SSE event generator.

    Phases:
        1. Setup (verify chat, load history, save user message)
        2. Fetch sources (parallel to stream start for low latency)
        3. Stream tokens — emit one SSE event per token
        4. Collect full answer, save to DB
        5. Emit sources event
        6. Emit done event

    On any error: emit error event and close stream cleanly.
    """
    chat_service = ChatService(db)

    try:
        # ── Phase 1: Setup ────────────────────────────────────────
        chat = await chat_service.get_chat(chat_id, current_user)
        history = await chat_service.load_history(chat_id)
        await chat_service.save_user_message(chat_id, query)

        # ── Phase 2: Pre-fetch sources (runs before stream starts) ─
        sources = await rag_service.get_sources_for_query(
            query=query,
            workspace_id=chat.workspace_id,
            user=current_user,
            document_ids=document_ids,
        )

        # ── Phase 3: Stream tokens ────────────────────────────────
        full_answer_parts: list[str] = []

        async for token in rag_service.stream(
            query=query,
            workspace_id=chat.workspace_id,
            user=current_user,
            history_messages=history,
            document_ids=document_ids,
        ):
            full_answer_parts.append(token)
            yield _sse_event({"type": "token", "content": token})

        # ── Phase 4: Persist assistant message ────────────────────
        full_answer = "".join(full_answer_parts)
        assistant_msg = await chat_service.save_assistant_message(
            chat_id=chat_id,
            content=full_answer,
            sources=sources,
        )
        await db.commit()

        # Auto-title on first exchange
        if chat.message_count <= 2:
            title = ChatService.generate_title(query)
            await chat_service.update_chat_title(chat_id, title, current_user)
            await db.commit()

        # ── Phase 5: Sources event ────────────────────────────────
        from src.schemas.chat import CitedSourceResponse
        yield _sse_event({
            "type": "sources",
            "sources": [
                {
                    "document_id": s.document_id,
                    "source": s.source_filename,
                    "page": s.page_number,
                    "score": s.relevance_score,
                }
                for s in sources
            ],
        })

        # ── Phase 6: Done event ───────────────────────────────────
        yield _sse_event({"type": "done", "message_id": str(assistant_msg.id)})

    except Exception as exc:
        from src.core.logger import get_logger
        log = get_logger(__name__)
        log.error("sse_stream_error", chat_id=str(chat_id), error=str(exc))
        yield _sse_event({"type": "error", "message": "An error occurred. Please try again."})


def _sse_event(data: dict) -> str:
    """
    Format a dict as a Server-Sent Event string.
    SSE format: "data: <json>\\n\\n"
    The double newline is the SSE message terminator.
    """
    return f"data: {json.dumps(data)}\n\n"