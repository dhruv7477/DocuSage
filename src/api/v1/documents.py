"""
src/api/v1/documents.py
────────────────────────
Document upload, listing, and management endpoints.

POST /documents/upload triggers the ingestion pipeline via Celery.
The API responds immediately with status=PENDING.
The client polls GET /documents/{id} to track progress.
"""

import asyncio
import mimetypes
import uuid
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.core.rate_limit import limiter, setting_limit
from src.db.session import get_db
from src.dependencies import get_current_user
from src.models.document import DocumentStatus, DocumentType
from src.models.user import User
from src.schemas.document import DocumentListResponse, DocumentResponse
from src.services.document_service import DocumentService
from src.services.quota_service import QuotaService
from src.services.storage_service import StorageService

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def _get_service(
    db: AsyncSession = Depends(get_db, scope="function"),
) -> DocumentService:
    return DocumentService(db, StorageService(), QuotaService(db))


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion (PDF, DOCX, TXT, MD, HTML)",
)
@limiter.limit(setting_limit("rate_limit_upload"))
async def upload_document(
    request: Request,
    workspace_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(_get_service),
) -> DocumentResponse:
    log.info(
        "upload_request_received",
        filename=file.filename,
        content_type=file.content_type,
        workspace_id=str(workspace_id),
        user_id=str(current_user.id),
    )

    content = await file.read()
    log.info("upload_file_read", filename=file.filename, bytes=len(content))

    document = await service.upload_document(
        workspace_id=workspace_id,
        user=current_user,
        filename=file.filename or "untitled",
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )
    log.info(
        "upload_stored",
        document_id=str(document.id),
        filename=file.filename,
        status=document.status.value,
    )

    # Dispatch Celery ingestion task
    try:
        from src.workers.tasks import ingest_document_task

        log.info("celery_dispatch_start", document_id=str(document.id))
        ingest_document_task.delay(str(document.id))
        log.info("celery_dispatch_ok", document_id=str(document.id))
    except Exception as exc:
        log.error(
            "celery_dispatch_failed", document_id=str(document.id), error=str(exc)
        )
        document.status = DocumentStatus.FAILED
        document.error_message = (
            "Failed to queue document for processing. Please try again."
        )
        await service._db.flush()

    return DocumentResponse.model_validate(document)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents uploaded by current user in a workspace",
)
async def list_documents(
    workspace_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(_get_service),
) -> DocumentListResponse:
    documents = await service.list_documents(workspace_id, current_user)
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in documents],
        total=len(documents),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document status and metadata",
)
async def get_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(_get_service),
) -> DocumentResponse:
    document = await service.get_document(document_id, current_user)
    return DocumentResponse.model_validate(document)


@router.get(
    "/{document_id}/file",
    summary="Stream the original uploaded file (for in-app PDF preview)",
    response_class=Response,
)
async def get_document_file(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(_get_service),
) -> Response:
    """
    Returns the raw bytes of the uploaded document with the correct
    Content-Type and an `inline` Content-Disposition so browsers render
    it directly (used by the SourcePanel PDF viewer).

    Auth: `service.get_document()` enforces user ownership — another
    user cannot fetch this file even if they guess the UUID.
    """
    document = await service.get_document(document_id, current_user)
    content = await service._storage.download(document.storage_path)

    media_type = (
        mimetypes.guess_type(document.original_filename)[0]
        or "application/octet-stream"
    )
    # RFC 5987 encoding so non-ASCII filenames survive the header.
    quoted = quote(document.original_filename)
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quoted}",
            "Cache-Control": "private, max-age=300",
        },
    )


@router.get(
    "/{document_id}/page-image",
    summary="Render one PDF page to PNG with the cited passage highlighted",
    response_class=Response,
)
@limiter.limit(setting_limit("rate_limit_page_image"))
async def get_document_page_image(
    request: Request,
    document_id: uuid.UUID,
    page: int = Query(..., ge=1, description="1-based page number"),
    chunk_id: str | None = Query(
        default=None, description="Chunk whose text to highlight"
    ),
    scale: float = Query(default=2.0, ge=0.5, le=3.0),
    crop: bool = Query(
        default=False,
        description="Clip to the highlighted passage instead of the full page",
    ),
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(_get_service),
) -> Response:
    """
    The "screenshot of the source" behind citation chips: the original
    PDF page rendered server-side (PyMuPDF) with amber highlight marks
    over the chunk text that grounded the answer.

    Headers: X-Page-Count, X-Highlight-Found ("1"/"0").
    422 for non-PDF documents or out-of-range pages; 404 for unowned
    documents/chunks.
    """
    document = await service.get_document(document_id, current_user)
    if document.file_type != DocumentType.PDF:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Page images are only available for PDF documents.",
        )

    highlight_text: str | None = None
    if chunk_id:
        from src.vector_store.factory import get_vector_store

        chunk = await get_vector_store().get_chunk(
            str(document.workspace_id), chunk_id, str(current_user.id)
        )
        if chunk is None or chunk.payload.document_id != str(document_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chunk not found for this document.",
            )
        highlight_text = chunk.payload.text

    pdf_bytes = await service._storage.download(document.storage_path)

    from src.services.page_renderer import render_pdf_page

    loop = asyncio.get_event_loop()
    try:
        rendered = await loop.run_in_executor(
            None, render_pdf_page, pdf_bytes, page, highlight_text, scale, crop
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    return Response(
        content=rendered.png,
        media_type="image/png",
        headers={
            "X-Page-Count": str(rendered.page_count),
            "X-Highlight-Found": "1" if rendered.highlight_found else "0",
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its vectors",
)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(_get_service),
) -> None:
    await service.delete_document(document_id, current_user)
