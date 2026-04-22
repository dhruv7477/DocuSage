"""
src/vector_store/qdrant_store.py
─────────────────────────────────
Qdrant implementation of BaseVectorStore.

User-level isolation design:
- Each workspace gets its own Qdrant collection:
    Collection name = "{prefix}_ws_{workspace_id}"
- Every point stores user_id in its payload
- EVERY search applies a mandatory Filter:
    must=[
        user_id == current_user.id,
        workspace_id == workspace_id
    ]
- Even if API auth is bypassed, the vector DB itself enforces isolation.
  This is defence-in-depth — two independent enforcement layers.

Collection configuration:
- Distance: COSINE (works with L2-normalised BGE vectors)
- Quantisation: off by default; enable scalar quantisation for large datasets
- On-disk payload indexing: enabled for user_id and document_id for fast filtering
"""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.configs.config import get_settings
from src.core.exceptions import VectorStoreError
from src.core.logger import get_logger
from src.vector_store.base import (
    BaseVectorStore,
    SearchFilter,
    SearchResult,
    VectorPayload,
)

log = get_logger(__name__)
settings = get_settings()


class QdrantVectorStore(BaseVectorStore):

    def __init__(self, url: str, api_key: str = "") -> None:
        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key or None,   # None disables auth (local dev)
            timeout=30,
        )

    # ── Collection Management ─────────────────────────────────────

    def _collection_name(self, workspace_id: str) -> str:
        return f"{settings.qdrant_collection_prefix}_ws_{workspace_id}"

    async def collection_exists(self, workspace_id: str) -> bool:
        try:
            await self._client.get_collection(self._collection_name(workspace_id))
            return True
        except Exception:
            return False

    async def create_collection(self, workspace_id: str, dimension: int) -> None:
        """
        Creates a Qdrant collection optimised for cosine similarity search.
        Adds payload indexes on user_id and document_id for fast filtering.
        Idempotent — safe to call if collection already exists.
        """
        name = self._collection_name(workspace_id)
        if await self.collection_exists(workspace_id):
            log.debug("collection_already_exists", collection=name)
            return

        try:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=dimension,
                    distance=qmodels.Distance.COSINE,
                    on_disk=False,   # Set True for large datasets to save RAM
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=10_000,  # Start HNSW indexing after 10K vectors
                ),
            )

            # ★ Index user_id for fast filtering — critical for isolation queries
            await self._client.create_payload_index(
                collection_name=name,
                field_name="user_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            # Index document_id for fast per-document deletion
            await self._client.create_payload_index(
                collection_name=name,
                field_name="document_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

            log.info(
                "collection_created",
                collection=name,
                dimension=dimension,
                workspace_id=workspace_id,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Failed to create Qdrant collection '{name}'",
                detail=str(exc),
            ) from exc

    async def delete_collection(self, workspace_id: str) -> None:
        name = self._collection_name(workspace_id)
        try:
            await self._client.delete_collection(name)
            log.info("collection_deleted", collection=name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete collection '{name}'", detail=str(exc)) from exc

    # ── Upsert ────────────────────────────────────────────────────

    async def upsert_vectors(
        self,
        workspace_id: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        payloads: list[VectorPayload],
    ) -> None:
        """
        Batch upsert vectors into the workspace collection.
        Qdrant handles duplicates via upsert semantics (insert or replace).
        """
        if not chunk_ids:
            return

        name = self._collection_name(workspace_id)
        points = [
            qmodels.PointStruct(
                id=cid,
                vector=vec,
                payload={
                    "user_id":      p.user_id,       # ★ Isolation key
                    "workspace_id": p.workspace_id,
                    "document_id":  p.document_id,
                    "chunk_index":  p.chunk_index,
                    "text":         p.text,
                    "source":       p.source,
                    "page_number":  p.page_number,
                    "chunk_total":  p.chunk_total,
                },
            )
            for cid, vec, p in zip(chunk_ids, vectors, payloads)
        ]

        try:
            await self._client.upsert(
                collection_name=name,
                points=points,
                wait=True,   # Synchronous — confirm write before returning
            )
            log.debug(
                "vectors_upserted",
                collection=name,
                count=len(points),
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Qdrant upsert failed for collection '{name}'",
                detail=str(exc),
            ) from exc

    # ── Search ────────────────────────────────────────────────────

    async def search(
        self,
        workspace_id: str,
        query_vector: list[float],
        search_filter: SearchFilter,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[SearchResult]:
        """
        Similarity search with MANDATORY user+workspace isolation.

        The filter ALWAYS includes user_id. This is not optional and
        cannot be bypassed by callers — it is hardcoded into this method.
        """
        name = self._collection_name(workspace_id)

        # ★ Build mandatory isolation filter
        must_conditions = [
            qmodels.FieldCondition(
                key="user_id",
                match=qmodels.MatchValue(value=search_filter.user_id),
            ),
            qmodels.FieldCondition(
                key="workspace_id",
                match=qmodels.MatchValue(value=search_filter.workspace_id),
            ),
        ]

        # Optional: filter to specific documents
        if search_filter.document_ids:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchAny(any=search_filter.document_ids),
                )
            )

        qdrant_filter = qmodels.Filter(must=must_conditions)

        try:
            results = await self._client.search(
                collection_name=name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Qdrant search failed for collection '{name}'",
                detail=str(exc),
            ) from exc

        search_results = []
        for hit in results:
            p = hit.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=str(hit.id),
                    score=hit.score,
                    payload=VectorPayload(
                        user_id=p.get("user_id", ""),
                        workspace_id=p.get("workspace_id", ""),
                        document_id=p.get("document_id", ""),
                        chunk_index=p.get("chunk_index", 0),
                        text=p.get("text", ""),
                        source=p.get("source", ""),
                        page_number=p.get("page_number", 0),
                        chunk_total=p.get("chunk_total", 0),
                    ),
                )
            )

        log.debug(
            "vector_search",
            collection=name,
            user_id=search_filter.user_id,
            results=len(search_results),
        )
        return search_results

    # ── Delete ────────────────────────────────────────────────────

    async def delete_document_vectors(
        self, workspace_id: str, document_id: str, user_id: str
    ) -> None:
        """
        Delete all vectors for a specific document.
        Filter includes user_id to prevent cross-user deletions.
        """
        name = self._collection_name(workspace_id)
        try:
            await self._client.delete(
                collection_name=name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchValue(value=document_id),
                            ),
                            qmodels.FieldCondition(
                                key="user_id",
                                match=qmodels.MatchValue(value=user_id),
                            ),
                        ]
                    )
                ),
                wait=True,
            )
            log.info(
                "document_vectors_deleted",
                document_id=document_id,
                workspace_id=workspace_id,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Failed to delete vectors for document '{document_id}'",
                detail=str(exc),
            ) from exc

    # ── Health ────────────────────────────────────────────────────

    async def health(self) -> bool:
        try:
            info = await self._client.get_collections()
            return info is not None
        except Exception:
            return False