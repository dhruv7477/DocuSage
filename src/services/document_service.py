"""
src/services/document_service.py
──────────────────────────────────
Document management service.

Responsibilities:
- Validate and store uploaded files
- Track document metadata in PostgreSQL
- Trigger ingestion pipeline (via Celery or direct)
- List/delete documents with ownership checks

The file is stored in MinIO/S3 BEFORE the Celery task runs.
This means the API responds quickly; ingestion is async.
"""

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs.config import get_settings
from src.core.exceptions import (
    DocumentAccessDeniedError,
    DocumentNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from src.core.logger import get_logger
from src.ingestion.parsers.registry import ParserRegistry
from src.models.document import Document, DocumentStatus, DocumentType
from src.models.user import User
from src.services.membership import assert_workspace_member
from src.services.quota_service import QuotaService
from src.services.storage_service import StorageService

log = get_logger(__name__)
settings = get_settings()

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB hard limit

_EXT_TO_TYPE: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".doc": DocumentType.DOCX,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MARKDOWN,
    ".markdown": DocumentType.MARKDOWN,
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
}


class DocumentService:
    def __init__(
        self,
        db: AsyncSession,
        storage: StorageService,
        quota: QuotaService | None = None,
    ) -> None:
        self._db = db
        self._storage = storage
        # Upload quotas apply only where a QuotaService is passed in (the API).
        self._quota = quota

    # ── Upload ────────────────────────────────────────────────────

    async def upload_document(
        self,
        workspace_id: uuid.UUID,
        user: User,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> Document:
        """
        Validate, store, and register a document for ingestion.

        Steps:
            1. Validate membership, file type, file size
            2. Upload to MinIO/S3
            3. Create Document record (status=PENDING) and commit it
            4. Return document — Celery task is dispatched separately

        The row is committed here, not at the end of the request, because
        the ingestion worker reads it in its own transaction. A task
        queued before the commit could start, find no row, and leave the
        document PENDING forever.

        Raises:
            WorkspaceAccessDeniedError: user not a member
            UnsupportedFileTypeError:   file extension not supported
            FileTooLargeError:          exceeds 50MB limit
            QuotaExceededError:         document or storage quota used up
        """
        # Validate workspace membership
        await self._assert_workspace_member(workspace_id, user.id)

        # Validate file type
        if not ParserRegistry.is_supported(filename):
            raise UnsupportedFileTypeError(
                f"'{Path(filename).suffix}' files are not supported.",
                detail=f"Supported: {', '.join(ParserRegistry.supported_extensions())}",
            )

        # Validate file size
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File exceeds 50MB limit ({len(content) / 1024 / 1024:.1f} MB uploaded)."
            )

        ext = Path(filename).suffix.lower()

        # Idempotent re-upload: reset existing record rather than creating a duplicate
        existing_result = await self._db.execute(
            select(Document).where(
                Document.workspace_id == workspace_id,
                Document.user_id == user.id,
                Document.original_filename == filename,
            )
        )
        existing = existing_result.scalar_one_or_none()

        if self._quota is not None:
            await self._quota.check_upload(
                user,
                new_bytes=len(content),
                replaces_bytes=existing.file_size_bytes if existing else None,
            )

        if existing is not None:
            doc_id = existing.id
            storage_path = existing.storage_path
            await self._storage.upload(content, storage_path, content_type)
            existing.status = DocumentStatus.PENDING
            existing.file_size_bytes = len(content)
            existing.chunk_count = 0
            existing.progress = 0
            existing.error_message = None
            await self._db.commit()
            log.info("document_re_uploaded", document_id=str(doc_id), filename=filename)
            return existing

        doc_id = uuid.uuid4()
        safe_filename = f"{doc_id}{ext}"

        # Build storage path
        storage_path = StorageService.build_path(
            workspace_id=str(workspace_id),
            user_id=str(user.id),
            document_id=str(doc_id),
            filename=safe_filename,
        )

        await self._storage.upload(content, storage_path, content_type)

        # If anything below this line raises, we must clean up the blob
        # we just put into MinIO/S3 — otherwise we leave an orphan that no
        # database row tracks. Failures here are rare (constraint violations,
        # session closed, etc.) but the cleanup is cheap.
        try:
            document = Document(
                id=doc_id,
                workspace_id=workspace_id,
                user_id=user.id,
                filename=safe_filename,
                original_filename=filename,
                file_type=_EXT_TO_TYPE.get(ext, DocumentType.TXT),
                file_size_bytes=len(content),
                storage_path=storage_path,
                status=DocumentStatus.PENDING,
            )
            self._db.add(document)
            await self._db.commit()
        except Exception:
            try:
                await self._storage.delete(storage_path)
            except Exception as cleanup_exc:
                log.warning(
                    "orphan_blob_cleanup_failed",
                    storage_path=storage_path,
                    error=str(cleanup_exc),
                )
            raise

        log.info(
            "document_uploaded",
            document_id=str(doc_id),
            filename=filename,
            user_id=str(user.id),
            workspace_id=str(workspace_id),
            size_kb=len(content) // 1024,
        )
        return document

    # ── List ──────────────────────────────────────────────────────

    async def list_documents(
        self, workspace_id: uuid.UUID, user: User
    ) -> list[Document]:
        """
        List documents owned by this user in this workspace.
        Users can ONLY see their own documents.
        """
        await self._assert_workspace_member(workspace_id, user.id)

        result = await self._db.execute(
            select(Document)
            .where(
                Document.workspace_id == workspace_id,
                Document.user_id == user.id,  # ★ User isolation
            )
            .order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())

    # ── Get ───────────────────────────────────────────────────────

    async def get_document(self, document_id: uuid.UUID, user: User) -> Document:
        """
        Fetch a single document. Verifies ownership.

        Raises:
            DocumentNotFoundError, DocumentAccessDeniedError
        """
        result = await self._db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
        if document.user_id != user.id:
            raise DocumentAccessDeniedError("You do not have access to this document.")

        return document

    # ── Delete ────────────────────────────────────────────────────

    async def delete_document(self, document_id: uuid.UUID, user: User) -> None:
        """
        Delete document record and storage file.
        Vector cleanup is handled by a Celery task triggered here.
        """
        document = await self.get_document(document_id, user)

        # Delete from storage (non-fatal if file missing)
        try:
            await self._storage.delete(document.storage_path)
        except Exception as exc:
            log.warning(
                "storage_delete_failed",
                path=document.storage_path,
                error=str(exc),
            )

        await self._db.delete(document)
        log.info(
            "document_deleted",
            document_id=str(document_id),
            user_id=str(user.id),
        )

        # Dispatch Celery task to clean up vectors
        # (imported here to avoid circular imports)
        try:
            from src.workers.tasks import delete_document_vectors_task

            delete_document_vectors_task.delay(
                workspace_id=str(document.workspace_id),
                document_id=str(document_id),
                user_id=str(user.id),
            )
        except Exception:
            log.warning(
                "vector_cleanup_task_dispatch_failed", document_id=str(document_id)
            )

    # ── Private ───────────────────────────────────────────────────

    async def _assert_workspace_member(
        self, workspace_id: uuid.UUID, user_id: uuid.UUID
    ) -> None:
        await assert_workspace_member(self._db, workspace_id, user_id)
