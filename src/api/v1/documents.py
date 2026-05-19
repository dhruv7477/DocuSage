"""
src/api/v1/documents.py
────────────────────────
Document upload, listing, and management endpoints.

POST /documents/upload triggers the ingestion pipeline via Celery.
The API responds immediately with status=PENDING.
The client polls GET /documents/{id} to track progress.
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.db.session import get_db
from src.dependencies import get_current_user
from src.models.document import DocumentStatus
from src.models.user import User
from src.schemas.document import DocumentListResponse, DocumentResponse
from src.services.document_service import DocumentService
from src.services.storage_service import StorageService

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def _get_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
    return DocumentService(db, StorageService())


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion (PDF, DOCX, TXT, MD, HTML)",
)
async def upload_document(
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
        log.error("celery_dispatch_failed", document_id=str(document.id), error=str(exc))
        document.status = DocumentStatus.FAILED
        document.error_message = "Failed to queue document for processing. Please try again."
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