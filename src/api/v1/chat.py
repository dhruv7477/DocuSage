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

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import LLMProviderError
from src.core.logger import get_logger
from src.core.rate_limit import QUESTION_SCOPE, chat_limit, limiter, setting_limit
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
from src.services.answer_validator import validate_answer
from src.services.chat_orchestrator import ChatOrchestrator, StreamPlan
from src.services.chat_service import ChatService
from src.services.quota_service import QuotaService
from src.services.doc_summary_service import DocSummaryService
from src.services.prompt_builder import PromptBuilder
from src.services.question_analyzer import QuestionAnalyzer
from src.services.rag_service import RAGService
from src.vector_store.factory import get_vector_store

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Dependency factories ──────────────────────────────────────────


def _get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    return ChatService(db)


def _get_rag_service() -> RAGService:
    """
    Wire the Phase 3 retrieval stack:

        QueryUnderstanding → HybridRetriever → Reranker → RAGService

    Replaces the legacy `RetrievalPipeline` (HyDE + multi-query +
    fake-BM25 + Jaccard dedup) with the focused, explainable pipeline
    designed in the redesign plan. All collaborators are stateless or
    cached singletons so this factory is cheap to call per request.
    """
    from src.configs.config import get_settings
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.query_understanding import QueryUnderstanding
    from src.retrieval.reranker import get_reranker

    _settings = get_settings()
    llm = get_llm_provider()
    embedder = get_embedding_provider()
    vector_store = get_vector_store()

    return RAGService(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        prompt_builder=PromptBuilder(
            max_context_tokens=_settings.rag_context_tokens,
            max_history_tokens=_settings.rag_history_tokens,
        ),
        retriever=HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            candidate_k=_settings.retrieval_candidate_k,
        ),
        reranker=get_reranker(
            strategy=_settings.reranker,
            llm=llm,
            model_name=_settings.reranker_model,
        ),
        query_understanding=QueryUnderstanding(
            llm=llm,
            enable_hyde=_settings.enable_hyde,
        ),
        top_k=_settings.rag_top_k,
        abstention_threshold=_settings.abstention_threshold,
    )


def _get_orchestrator(
    rag_service: RAGService = Depends(_get_rag_service),
) -> ChatOrchestrator:
    """
    Compose the agentic pipeline: QuestionAnalyzer (router) +
    RAGService + DocSummaryService. Patched wholesale in tests.
    """
    from src.configs.config import get_settings
    from src.providers.llm.factory import (
        get_router_llm_provider,
        get_summary_llm_provider,
    )

    _settings = get_settings()
    return ChatOrchestrator(
        analyzer=QuestionAnalyzer(
            # Small dedicated classifier model, not the 7B answer model.
            get_router_llm_provider(),
            max_tokens=_settings.router_max_tokens,
            use_llm=_settings.router_use_llm,
            llm_timeout_seconds=_settings.router_timeout_seconds,
        ),
        rag_service=rag_service,
        # Summaries run on a lighter model (see get_summary_llm_provider):
        # the answer model is too memory/compute-heavy for whole-document
        # map-reduce on CPU and was timing out.
        summary_service=DocSummaryService(
            get_vector_store(), get_summary_llm_provider()
        ),
        enable_router=_settings.enable_question_router,
    )


# ── Session CRUD ─────────────────────────────────────────────────


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
@limiter.limit(setting_limit("rate_limit_chat_create"))
async def create_chat(
    request: Request,
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
    chats = await chat_service.list_chats(
        workspace_id, current_user, limit=limit, offset=offset
    )
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
@limiter.shared_limit(chat_limit, scope=QUESTION_SCOPE)
async def query(
    request: Request,
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

    # Verify ownership and current membership, then the daily question limits
    chat = await chat_service.get_chat_for_question(chat_id, current_user)
    await QuotaService(db).reserve_question(current_user)

    # Load history for context
    history = await chat_service.load_history(chat_id)

    # Persist user message
    await chat_service.save_user_message(chat_id, payload.query)

    # Run RAG pipeline. Note: the question router only runs on the SSE
    # path (the product path) — this endpoint is for programmatic
    # access where callers want deterministic RAG behavior.
    rag_response = await rag_service.generate(
        query=payload.query,
        workspace_id=chat.workspace_id,
        user=current_user,
        history_messages=history,
        document_ids=payload.document_ids,
    )

    # Programmatic validation: strip hallucinated [n] markers, enforce
    # the abstention contract. Sources here come from the dedup'd view,
    # so validate against the prompt's included_results count when we
    # have it (that's what the [n] numbering refers to).
    num_sources = (
        len(rag_response.prompt.included_results)
        if rag_response.prompt is not None
        else len(rag_response.sources)
    )
    validation = validate_answer(
        rag_response.answer,
        num_sources=num_sources,
        abstained=rag_response.abstained,
        allow_citations=True,
    )
    rag_response.answer = validation.answer

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
                chunk_id=s.chunk_id,
                heading_path=s.heading_path,
                char_start=s.char_start,
                char_end=s.char_end,
                text=s.text,
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
@limiter.shared_limit(chat_limit, scope=QUESTION_SCOPE)
async def stream_query(
    request: Request,
    chat_id: uuid.UUID,
    query: str = Query(..., min_length=1, max_length=4000),
    document_ids: str | None = Query(
        default=None, description="Comma-separated document IDs to scope search"
    ),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    orchestrator: ChatOrchestrator = Depends(_get_orchestrator),
) -> StreamingResponse:
    """
    SSE streaming endpoint for real-time chat UI.

    Protocol (full event contract):
        data: {"type":"thinking","stage":"analyzing|routing|retrieving|reranking|generating|summarizing|validating"}
        data: {"type":"route","intent":"rag_query|doc_summary|off_topic"}
        data: {"type":"token","content":"<text>"}        (repeated)
        data: {"type":"answer_final","content":"<text>"} (only if the validator corrected the answer)
        data: {"type":"warning","code":"...","message":"...","confidence":...}
        data: {"type":"sources","sources":[...]}         (context-ordered: sources[n-1] backs citation [n])
        data: {"type":"done","message_id":"<id>"}        (final event)

    Connect using fetchEventSource with Authorization header:
        fetchEventSource(`/api/v1/chat/${chatId}/stream?query=...`, {
            headers: { Authorization: `Bearer ${token}` }
        })
    """
    doc_id_list = [d.strip() for d in document_ids.split(",")] if document_ids else None

    # Access and quota errors must be a normal JSON error response, so they
    # are checked before the stream starts (afterwards the status is 200).
    await ChatService(db).get_chat_for_question(chat_id, current_user)
    await QuotaService(db).reserve_question(current_user)

    return StreamingResponse(
        _sse_generator(
            chat_id=chat_id,
            query=query,
            document_ids=doc_id_list,
            current_user=current_user,
            db=db,
            orchestrator=orchestrator,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE
            "Connection": "keep-alive",
        },
    )


async def _sse_generator(
    chat_id: uuid.UUID,
    query: str,
    document_ids: list[str] | None,
    current_user: User,
    db: AsyncSession,
    orchestrator: ChatOrchestrator,
) -> AsyncIterator[str]:
    """
    Core SSE event generator for the agentic pipeline.

    Phases:
        1. Setup (verify chat, load history, save user message, list
           the workspace's completed documents for the router)
        2. Orchestrate + stream — a background task runs
           `orchestrator.prepare()` (analyze → route → worker) and then
           drains the route's token iterator. Stage events flow through
           the same queue; the consumer emits a keepalive heartbeat with
           the *current* stage every 8 s of silence.
        3. Validate the finished answer (programmatic: citation mapping,
           abstention contract) and persist the corrected text
        4. `warning` events (abstention / validator findings)
        5. `sources` event — context-ordered: sources[n-1] backs [n]
        6. `done` event

    On any error: emit error event and close stream cleanly.
    """
    chat_service = ChatService(db)
    log = get_logger(__name__)

    producer_task: asyncio.Task | None = None
    try:
        # ── Phase 1: Setup ────────────────────────────────────────
        chat = await chat_service.get_chat(chat_id, current_user)
        history = await chat_service.load_history(chat_id)

        # Idempotency guard: a reconnect/retry of this GET re-sends the
        # same query — don't persist the user message twice in a row.
        last = history[-1] if history else None
        is_duplicate_resend = (
            last is not None
            and str(getattr(last, "role", "")) in ("user", "MessageRole.USER")
            and getattr(last, "content", None) == query
        )
        if not is_duplicate_resend:
            await chat_service.save_user_message(chat_id, query)
            # Commit the user turn before generation begins. Otherwise it
            # shares a transaction with the assistant message (committed only
            # at the end), so any LLM failure rolls the user message back and
            # it vanishes from the thread on the next refetch.
            await db.commit()
        else:
            log.info("duplicate_user_message_skipped", chat_id=str(chat_id))

        # Completed documents in this workspace — shown to the router so
        # it can resolve "summarize the I-9 form" to a document id.
        from src.models.document import DocumentStatus
        from src.services.document_service import DocumentService
        from src.services.storage_service import StorageService

        docs = await DocumentService(db, StorageService()).list_documents(
            chat.workspace_id, current_user
        )
        workspace_documents = [
            (str(d.id), d.original_filename)
            for d in docs
            if d.status == DocumentStatus.COMPLETED
        ]

        # ── Phase 2: Orchestrate + stream w/ keepalive ────────────
        _KEEPALIVE_INTERVAL = 8.0  # seconds between heartbeats
        token_queue: asyncio.Queue[dict | BaseException | None] = asyncio.Queue()
        plan_box: list[StreamPlan] = []

        def _emit(event: dict) -> None:
            # Called from the producer task on this same event loop.
            token_queue.put_nowait({"kind": "event", "event": event})

        async def _produce() -> None:
            try:
                plan = await orchestrator.prepare(
                    query=query,
                    workspace_id=chat.workspace_id,
                    user=current_user,
                    history_messages=history,
                    document_ids=document_ids,
                    workspace_documents=workspace_documents,
                    emit=_emit,
                )
                plan_box.append(plan)
                async for tok in plan.token_iter:
                    await token_queue.put({"kind": "token", "text": tok})
                await token_queue.put(None)  # sentinel: stream finished cleanly
            except BaseException as exc:
                await token_queue.put(exc)  # forward any error to the consumer

        producer_task = asyncio.create_task(_produce())

        full_answer_parts: list[str] = []
        stage = "analyzing"
        yield _sse_event({"type": "thinking", "stage": stage})

        while True:
            try:
                item = await asyncio.wait_for(
                    token_queue.get(), timeout=_KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                # Nothing arrived in 8 s — heartbeat the current stage.
                yield _sse_event({"type": "thinking", "stage": stage})
                continue

            if item is None:
                break  # Stream completed
            if isinstance(item, BaseException):
                raise item  # Re-raise so the outer except handles it

            if item["kind"] == "event":
                event = item["event"]
                if event.get("type") == "thinking":
                    stage = event.get("stage", stage)
                yield _sse_event(event)
                continue

            # Token item
            if not full_answer_parts and stage != "generating":
                stage = "generating"
                yield _sse_event({"type": "thinking", "stage": stage})
            full_answer_parts.append(item["text"])
            yield _sse_event({"type": "token", "content": item["text"]})

        plan = plan_box[0] if plan_box else None
        cited = plan.sources if plan else []
        confidence = plan.confidence if plan else 0.0
        abstained = plan.abstained if plan else True
        intent = plan.intent if plan else "rag_query"

        # Source citations belong to grounded RAG answers only. Summaries
        # (and off-topic replies) are not per-passage citable, so we neither
        # persist nor surface sources for them — the UI shows no "sources".
        display_sources = cited if intent == "rag_query" else []

        # ── Phase 3: Validate + persist ───────────────────────────
        yield _sse_event({"type": "thinking", "stage": "validating"})
        result = validate_answer(
            "".join(full_answer_parts),
            num_sources=len(cited),
            abstained=abstained,
            allow_citations=plan.allow_citations if plan else False,
        )
        if result.changed:
            # Frontend replaces the streamed text wholesale.
            yield _sse_event({"type": "answer_final", "content": result.answer})

        assistant_msg = await chat_service.save_assistant_message(
            chat_id=chat_id,
            content=result.answer,
            sources=display_sources,
        )

        if chat.message_count <= 2:
            title = ChatService.generate_title(query)
            await chat_service.update_chat_title(chat_id, title, current_user)

        await db.commit()

        # ── Phase 4: Warning events ───────────────────────────────
        if abstained and intent == "rag_query":
            code = "no_documents" if not cited else "low_confidence"
            yield _sse_event(
                {
                    "type": "warning",
                    "code": code,
                    "message": (
                        "No matching content found in your documents."
                        if code == "no_documents"
                        else "The retrieved context is weak — the answer may be incomplete."
                    ),
                    "confidence": round(confidence, 3),
                }
            )
        if result.warning_code == "model_abstained":
            yield _sse_event(
                {
                    "type": "warning",
                    "code": "model_abstained",
                    "message": (
                        "The model declined to answer even though related "
                        "content was found — try rephrasing the question."
                    ),
                    "confidence": round(confidence, 3),
                }
            )

        # ── Phase 5: Sources event (context-ordered; RAG only) ────
        yield _sse_event(
            {
                "type": "sources",
                "sources": [
                    {
                        "document_id": s.document_id,
                        "chunk_id": s.chunk_id,
                        "source": s.source_filename,
                        "page": s.page_number,
                        "score": s.relevance_score,
                        "heading_path": s.heading_path,
                        "char_start": s.char_start,
                        "char_end": s.char_end,
                        "text": s.text,
                    }
                    for s in display_sources
                ],
                "confidence": round(confidence, 3),
            }
        )

        # ── Phase 6: Done event ───────────────────────────────────
        yield _sse_event({"type": "done", "message_id": str(assistant_msg.id)})

    except Exception as exc:
        log.error(
            "sse_stream_error", chat_id=str(chat_id), error=str(exc), exc_info=True
        )
        # LLMProviderError carries a user-safe message (e.g. "the model timed
        # out — it may still be loading"); surface it so the user knows whether
        # to retry. Fall back to a generic message for unexpected errors.
        user_message = (
            exc.message
            if isinstance(exc, LLMProviderError)
            else "Something went wrong. Please try again."
        )
        yield _sse_event(
            {
                "type": "error",
                "message": user_message,
            }
        )
    finally:
        # Cancel the producer task if the client disconnected or an error
        # occurred before the stream finished.
        if producer_task is not None and not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass


def _sse_event(data: dict) -> str:
    """
    Format a dict as a Server-Sent Event string.
    SSE format: "data: <json>\\n\\n"
    The double newline is the SSE message terminator.
    """
    return f"data: {json.dumps(data)}\n\n"
