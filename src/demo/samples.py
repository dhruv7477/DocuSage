"""
src/demo/samples.py
───────────────────
Sample documents and nightly cleanup for the public demo (DEMO_MODE=true).

How it works:
- The sample library is ingested once. A system user that cannot sign in
  owns a hidden "Sample library" workspace. Each PDF in DEMO_SAMPLES_DIR
  becomes one of its documents and goes through the normal ingestion
  pipeline.
- Every new account gets its own "Sample documents" workspace. Its
  document rows point at the library's stored files, and the library's
  vectors are copied into the account's collection with the account's IDs,
  so nothing is embedded again and the usual isolation filters apply.
- Every night, every document that is not a sample is deleted: its
  vectors, its stored file and its row.

A document counts as a sample when its storage path is one of the
library's. Deleting a sample copy never deletes the shared file (see
DocumentService.delete_document).
"""

import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.configs.config import get_settings
from src.core.logger import get_logger
from src.models.document import Document, DocumentStatus, DocumentType
from src.models.user import User, UserRole
from src.models.workspace import Workspace, WorkspaceMember, WorkspaceMemberRole
from src.services.storage_service import StorageService
from src.vector_store.base import BaseVectorStore, DocumentRef

log = get_logger(__name__)

LIBRARY_EMAIL = "sample-library@docusage.invalid"
LIBRARY_WORKSPACE = "Sample library"
USER_WORKSPACE = "Sample documents"
USER_WORKSPACE_DESCRIPTION = "Public-domain documents to try DocuSage with."


class SeedOutcome(str, Enum):
    SEEDED = "seeded"
    ALREADY_SEEDED = "already_seeded"
    LIBRARY_NOT_READY = "library_not_ready"
    USER_NOT_FOUND = "user_not_found"


@dataclass
class Library:
    owner: User
    workspace: Workspace
    documents: list[Document]


# ── Library ───────────────────────────────────────────────────────


def sample_files() -> list[Path]:
    """PDF files in DEMO_SAMPLES_DIR, sorted by name."""
    folder = Path(get_settings().demo_samples_dir)
    if not folder.is_dir():
        log.warning("demo_samples_dir_missing", path=str(folder))
        return []
    return sorted(folder.glob("*.pdf"))


async def _library_owner_and_workspace(db: AsyncSession) -> tuple[User, Workspace]:
    owner = await db.scalar(select(User).where(User.email == LIBRARY_EMAIL))
    if owner is None:
        owner = User(
            email=LIBRARY_EMAIL,
            password_hash=None,
            full_name="Sample library",
            role=UserRole.MEMBER,
            is_active=False,  # cannot sign in
        )
        db.add(owner)
        await db.flush()
    workspace = await db.scalar(
        select(Workspace).where(
            Workspace.owner_id == owner.id, Workspace.name == LIBRARY_WORKSPACE
        )
    )
    if workspace is None:
        workspace = Workspace(name=LIBRARY_WORKSPACE, owner_id=owner.id)
        db.add(workspace)
        await db.flush()
    return owner, workspace


async def load_library(db: AsyncSession) -> Library:
    """The library as it is now (its documents may still be ingesting)."""
    owner, workspace = await _library_owner_and_workspace(db)
    docs = await db.scalars(
        select(Document)
        .where(Document.workspace_id == workspace.id, Document.user_id == owner.id)
        .order_by(Document.original_filename)
    )
    return Library(owner, workspace, list(docs))


async def ensure_library(db: AsyncSession, storage: StorageService) -> list[str]:
    """Register every sample file in the library; return document IDs to ingest.

    Idempotent. A new file is stored and registered, and a FAILED one is
    queued again. Files already registered and in flight or completed are
    left alone. The caller commits, then queues the returned documents.
    """
    library = await load_library(db)
    by_name = {d.original_filename: d for d in library.documents}
    to_ingest: list[str] = []
    for path in sample_files():
        doc = by_name.get(path.name)
        if doc is not None and doc.status != DocumentStatus.FAILED:
            continue
        if doc is None:
            doc_id = uuid.uuid4()
            storage_path = StorageService.build_path(
                workspace_id=str(library.workspace.id),
                user_id=str(library.owner.id),
                document_id=str(doc_id),
                filename=f"{doc_id}{path.suffix.lower()}",
            )
            content = path.read_bytes()
            await storage.upload(content, storage_path, "application/pdf")
            doc = Document(
                id=doc_id,
                workspace_id=library.workspace.id,
                user_id=library.owner.id,
                filename=Path(storage_path).name,
                original_filename=path.name,
                file_type=DocumentType.PDF,
                file_size_bytes=len(content),
                storage_path=storage_path,
                status=DocumentStatus.PENDING,
            )
            db.add(doc)
        else:
            doc.status = DocumentStatus.PENDING
            doc.error_message = None
            doc.progress = 0
        await db.flush()
        to_ingest.append(str(doc.id))
    log.info(
        "sample_library_checked", files=len(sample_files()), to_ingest=len(to_ingest)
    )
    return to_ingest


# ── Per-account copies ────────────────────────────────────────────


async def seed_user(
    db: AsyncSession, vector_store: BaseVectorStore, user_id: uuid.UUID
) -> SeedOutcome:
    """Give one account its own copies of every sample document.

    Idempotent and safe to retry: a copy that did not finish is copied
    again. Commits as it goes, so finished copies are visible at once.
    """
    user = await db.get(User, user_id)
    if user is None:
        return SeedOutcome.USER_NOT_FOUND
    library = await load_library(db)
    names = {p.name for p in sample_files()}
    ready = [
        d
        for d in library.documents
        if d.original_filename in names and d.status == DocumentStatus.COMPLETED
    ]
    if not names or len(ready) < len(names):
        return SeedOutcome.LIBRARY_NOT_READY

    workspace = await db.scalar(
        select(Workspace).where(
            Workspace.owner_id == user.id, Workspace.name == USER_WORKSPACE
        )
    )
    if workspace is None:
        workspace = Workspace(
            name=USER_WORKSPACE,
            description=USER_WORKSPACE_DESCRIPTION,
            owner_id=user.id,
        )
        db.add(workspace)
        await db.flush()
        db.add(
            WorkspaceMember(
                workspace_id=workspace.id,
                user_id=user.id,
                role=WorkspaceMemberRole.ADMIN,
            )
        )
        await db.commit()

    copies = {
        d.storage_path: d
        for d in await db.scalars(
            select(Document).where(
                Document.workspace_id == workspace.id, Document.user_id == user.id
            )
        )
    }
    copied_any = False
    for sample in ready:
        copy = copies.get(sample.storage_path)
        if copy is not None and copy.status == DocumentStatus.COMPLETED:
            continue
        if copy is None:
            copy = Document(
                workspace_id=workspace.id,
                user_id=user.id,
                filename=sample.filename,
                original_filename=sample.original_filename,
                file_type=sample.file_type,
                file_size_bytes=sample.file_size_bytes,
                storage_path=sample.storage_path,  # shared, never deleted
                status=DocumentStatus.INDEXING,
                progress=80,
            )
            db.add(copy)
            await db.commit()
        await _copy_vectors(vector_store, library, sample, copy)
        await db.commit()
        copied_any = True

    log.info("samples_seeded", user_id=str(user.id), documents=len(ready))
    return SeedOutcome.SEEDED if copied_any else SeedOutcome.ALREADY_SEEDED


async def refresh_sample_copies(db: AsyncSession, vector_store: BaseVectorStore) -> int:
    """Copy the library's vectors again into every account's sample copies.

    For after the library is re-ingested by newer code (a change to how
    documents are chunked): accounts that already exist get the new chunks
    too. Uploads are not touched. Returns the number of copies refreshed, or
    0 when the library is not fully ingested.
    """
    library = await load_library(db)
    ready = {
        d.storage_path: d
        for d in library.documents
        if d.status == DocumentStatus.COMPLETED
    }
    if not ready or len(ready) < len(library.documents):
        log.warning("sample_copies_not_refreshed", reason="library not ready")
        return 0
    copies = list(
        await db.scalars(
            select(Document).where(
                Document.storage_path.in_(list(ready)),
                Document.user_id != library.owner.id,
            )
        )
    )
    for copy in copies:
        await _copy_vectors(vector_store, library, ready[copy.storage_path], copy)
        await db.commit()
    log.info("sample_copies_refreshed", copies=len(copies))
    return len(copies)


async def _copy_vectors(
    vector_store: BaseVectorStore, library: Library, sample: Document, copy: Document
) -> None:
    """Replace `copy`'s vectors with the library's for `sample`; mark it complete."""
    target = DocumentRef(str(copy.workspace_id), str(copy.id), str(copy.user_id))
    await vector_store.delete_document_vectors(
        target.workspace_id, target.document_id, target.user_id
    )
    count = await vector_store.copy_document_vectors(
        DocumentRef(str(library.workspace.id), str(sample.id), str(library.owner.id)),
        target,
    )
    copy.status = DocumentStatus.COMPLETED
    copy.chunk_count = count
    copy.progress = 100
    copy.error_message = None


# ── Nightly cleanup ───────────────────────────────────────────────


def library_storage_paths():  # type: ignore[no-untyped-def]
    """Subquery of the library's storage paths, which every sample copy shares."""
    return (
        select(Document.storage_path)
        .join(User, Document.user_id == User.id)
        .where(User.email == LIBRARY_EMAIL)
    )


async def wipe_user_uploads(
    db: AsyncSession, storage: StorageService, vector_store: BaseVectorStore
) -> int:
    """Delete every document that is not a sample. Returns how many were deleted.

    A document whose vectors or file cannot be removed is logged and kept,
    so the next run tries it again.
    """
    library = await load_library(db)
    keep = {d.storage_path for d in library.documents}
    stmt = select(Document).where(Document.user_id != library.owner.id)
    if keep:
        stmt = stmt.where(Document.storage_path.not_in(keep))
    uploads = await db.scalars(stmt)
    deleted = 0
    for doc in list(uploads):
        try:
            await vector_store.delete_document_vectors(
                str(doc.workspace_id), str(doc.id), str(doc.user_id)
            )
            await storage.delete(doc.storage_path)
        except Exception as exc:
            log.warning(
                "demo_wipe_document_failed", document_id=str(doc.id), error=str(exc)
            )
            continue
        await db.delete(doc)
        deleted += 1
    await db.commit()
    log.info("demo_uploads_wiped", deleted=deleted)
    return deleted
