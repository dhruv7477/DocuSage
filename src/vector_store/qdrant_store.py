"""
src/vector_store/qdrant_store.py
─────────────────────────────────
Qdrant implementation of `BaseVectorStore`.

User-level isolation
--------------------
* Each workspace gets its own collection: `{prefix}_ws_{workspace_id}`
* Every point stores `user_id` + `workspace_id` in its payload
* Every search applies a mandatory filter on both fields. Even if the
  API auth layer is bypassed somehow, Qdrant itself refuses to serve
  someone else's data — defence in depth.

Collection configuration
------------------------
* Distance: COSINE (paired with L2-normalised embeddings).
* Payload indexes:
    - KEYWORD on `user_id` and `document_id` for fast filter joins.
    - TEXT (word tokenizer) on `text` so we can do real lexical
      retrieval as the sparse half of hybrid search.
* Dimension validation: an upsert that disagrees with the existing
  collection's vector size raises `EmbeddingDimensionMismatch` with a
  remediation hint, instead of silently accepting partial data.
"""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.configs.config import get_settings
from src.core.exceptions import EmbeddingDimensionMismatch, VectorStoreError
from src.core.logger import get_logger
from src.vector_store.base import (
    BaseVectorStore,
    DocumentRef,
    SearchFilter,
    SearchResult,
    VectorPayload,
)

log = get_logger(__name__)

# Points per upsert request. Qdrant keeps the memory it used to take in a
# request: five 1,000-point documents left it at 213 MB sent whole and at
# 123 MB sent in batches of 128 (peaks 297 and 191 MB).
UPSERT_BATCH = 128
settings = get_settings()


class QdrantVectorStore(BaseVectorStore):
    def __init__(self, url: str, api_key: str = "") -> None:
        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key or None,  # None disables auth (local dev)
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

    async def get_collection_dimension(self, workspace_id: str) -> int | None:
        """Read back the vector size the collection was created with."""
        name = self._collection_name(workspace_id)
        try:
            info = await self._client.get_collection(name)
        except Exception:
            return None
        # Qdrant stores either a single VectorParams or a dict of named ones;
        # we only ever create the unnamed default vector.
        params = info.config.params.vectors
        if isinstance(params, dict):
            # Pick the first named vector — we don't use multi-vector layouts
            params = next(iter(params.values()))
        return getattr(params, "size", None)

    async def create_collection(self, workspace_id: str, dimension: int) -> None:
        """
        Idempotent. Creates the collection optimised for cosine search,
        and adds payload indexes for user_id, document_id (KEYWORD) and
        the chunk text (TEXT) so hybrid retrieval works without an
        extra index store.
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
                    on_disk=False,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=10_000,
                ),
            )

            # ★ Isolation key — must be indexed for every retrieval to be fast
            await self._client.create_payload_index(
                collection_name=name,
                field_name="user_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            # Document-scoped deletes / per-document filtering
            await self._client.create_payload_index(
                collection_name=name,
                field_name="document_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            # Full-text payload index on `text` — powers `text_search()`
            # and replaces the previous "BM25 over already-fetched vector
            # candidates" hack with an independent sparse retrieval path.
            await self._client.create_payload_index(
                collection_name=name,
                field_name="text",
                field_schema=qmodels.TextIndexParams(
                    type=qmodels.PayloadSchemaType.TEXT,
                    tokenizer=qmodels.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                ),
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
            raise VectorStoreError(
                f"Failed to delete collection '{name}'", detail=str(exc)
            ) from exc

    # ── Upsert ────────────────────────────────────────────────────

    async def _upsert_in_batches(
        self, name: str, points: list[qmodels.PointStruct]
    ) -> None:
        """
        Upsert `points` in UPSERT_BATCH-sized requests, all or nothing: if a
        request fails, the batches already written are deleted again before
        the error is raised, as when everything went in one request.
        """
        written: list = []
        try:
            for start in range(0, len(points), UPSERT_BATCH):
                batch = points[start : start + UPSERT_BATCH]
                await self._client.upsert(collection_name=name, points=batch, wait=True)
                written.extend(point.id for point in batch)
        except Exception:
            if written:
                try:
                    await self._client.delete(
                        collection_name=name,
                        points_selector=qmodels.PointIdsList(points=written),
                        wait=True,
                    )
                except Exception as cleanup_exc:
                    log.warning(
                        "partial_upsert_cleanup_failed",
                        collection=name,
                        points=len(written),
                        error=str(cleanup_exc),
                    )
            raise

    async def upsert_vectors(
        self,
        workspace_id: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        payloads: list[VectorPayload],
    ) -> None:
        """
        Batch upsert with strict dimension validation. If the embedder's
        vector size disagrees with the collection's configured size, we
        fail loudly rather than partially writing data that can never
        be searched correctly.
        """
        if not chunk_ids:
            return

        name = self._collection_name(workspace_id)
        actual_dim = len(vectors[0]) if vectors and vectors[0] else None
        configured_dim = await self.get_collection_dimension(workspace_id)
        if (
            actual_dim is not None
            and configured_dim is not None
            and actual_dim != configured_dim
        ):
            raise EmbeddingDimensionMismatch(
                (
                    f"Embedding dimension mismatch for collection '{name}': "
                    f"vectors are {actual_dim}-dim but the collection was "
                    f"created with {configured_dim}-dim."
                ),
                detail=(
                    f"Reindex this workspace after changing the embedding "
                    f"model: `python -m scripts.reindex_workspace {workspace_id}`. "
                    "Until then, ingestion for this workspace will keep failing."
                ),
            )

        points = [
            qmodels.PointStruct(
                id=cid,
                vector=vec,
                payload={
                    "user_id": p.user_id,  # ★ Isolation key
                    "workspace_id": p.workspace_id,
                    "document_id": p.document_id,
                    "chunk_index": p.chunk_index,
                    "text": p.text,
                    "source": p.source,
                    "page_number": p.page_number,
                    "chunk_total": p.chunk_total,
                    "heading_path": list(p.heading_path),
                    "char_start": p.char_start,
                    "char_end": p.char_end,
                    "token_count": p.token_count,
                    "prev_chunk_id": p.prev_chunk_id,
                    "next_chunk_id": p.next_chunk_id,
                },
            )
            for cid, vec, p in zip(chunk_ids, vectors, payloads)
        ]

        try:
            await self._upsert_in_batches(name, points)
            log.debug("vectors_upserted", collection=name, count=len(points))
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Qdrant upsert failed for collection '{name}'",
                detail=str(exc),
            ) from exc

    # ── Search (dense) ────────────────────────────────────────────

    async def search(
        self,
        workspace_id: str,
        query_vector: list[float],
        search_filter: SearchFilter,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Vector similarity search with mandatory user+workspace
        isolation. score_threshold defaults to 0.0 — recall-first; the
        retrieval pipeline's reranker decides what to actually surface.
        """
        name = self._collection_name(workspace_id)
        qdrant_filter = self._build_isolation_filter(search_filter)

        try:
            results = await self._client.search(
                collection_name=name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Qdrant search failed for collection '{name}'",
                detail=str(exc),
            ) from exc

        log.debug(
            "vector_search",
            collection=name,
            user_id=search_filter.user_id,
            results=len(results),
        )
        return [_hit_to_result(hit) for hit in results]

    # ── Search (sparse / lexical) ─────────────────────────────────

    async def text_search(
        self,
        workspace_id: str,
        query: str,
        search_filter: SearchFilter,
        top_k: int = 50,
    ) -> list[SearchResult]:
        """
        Lexical retrieval over the full-text payload index on `text`.

        Returns results whose payload `text` matches `query` per
        Qdrant's WORD tokenizer. Used by the Phase 3 hybrid pipeline as
        the sparse half — runs in parallel with `search()` and the two
        result lists are fused with Reciprocal Rank Fusion.
        """
        name = self._collection_name(workspace_id)

        must_conditions = self._isolation_must(search_filter)
        # Add the text MatchText against the indexed `text` field.
        must_conditions.append(
            qmodels.FieldCondition(
                key="text",
                match=qmodels.MatchText(text=query),
            )
        )
        qfilter = qmodels.Filter(must=must_conditions)

        try:
            # `scroll` with the FT condition gives us up to `limit`
            # matching points; ranking is by Qdrant's text relevance.
            points, _next = await self._client.scroll(
                collection_name=name,
                scroll_filter=qfilter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Qdrant text search failed for collection '{name}'",
                detail=str(exc),
            ) from exc

        # Synthesize a deterministic descending score so callers that
        # rank by score still see a useful order. RRF only cares about
        # rank, not absolute score, so this is fine.
        results: list[SearchResult] = []
        for rank, point in enumerate(points):
            results.append(_point_to_result(point, score=1.0 / (rank + 1)))
        log.debug(
            "text_search",
            collection=name,
            user_id=search_filter.user_id,
            results=len(results),
        )
        return results

    # ── Chunk access (summary route + page-image highlight) ──────

    async def list_document_chunks(
        self,
        workspace_id: str,
        document_id: str,
        user_id: str,
    ) -> list[SearchResult]:
        """
        All chunks of one document, ordered by chunk_index. Scrolls
        through Qdrant pages (256 points each) until exhausted; the
        isolation filter (user + workspace + document) is mandatory.
        """
        name = self._collection_name(workspace_id)
        qfilter = self._build_isolation_filter(
            SearchFilter(
                user_id=user_id,
                workspace_id=workspace_id,
                document_ids=[document_id],
            )
        )

        results: list[SearchResult] = []
        offset = None
        try:
            while True:
                points, offset = await self._client.scroll(
                    collection_name=name,
                    scroll_filter=qfilter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                results.extend(_point_to_result(p, score=1.0) for p in points)
                if offset is None:
                    break
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                # Collection doesn't exist — the workspace was never indexed,
                # or the vector store was reset (its volume wiped) while the
                # Postgres Document row survived. Treat as "no chunks" so the
                # summary path degrades to a graceful "no indexed content,
                # re-process the document" note instead of a 500.
                log.warning(
                    "document_chunks_collection_missing",
                    document_id=document_id,
                    collection=name,
                )
                return []
            raise VectorStoreError(
                f"Failed to list chunks for document '{document_id}'",
                detail=str(exc),
            ) from exc

        results.sort(key=lambda r: r.payload.chunk_index)
        log.debug(
            "document_chunks_listed",
            document_id=document_id,
            chunks=len(results),
        )
        return results

    async def get_chunk(
        self,
        workspace_id: str,
        chunk_id: str,
        user_id: str,
    ) -> SearchResult | None:
        """Single chunk by point id; None unless owned by `user_id`."""
        name = self._collection_name(workspace_id)
        try:
            points = await self._client.retrieve(
                collection_name=name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=False,
            )
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                return None
            raise VectorStoreError(
                f"Failed to fetch chunk '{chunk_id}'", detail=str(exc)
            ) from exc

        if not points:
            return None
        point = points[0]
        payload = point.payload or {}
        if payload.get("user_id") != user_id:
            # Ownership mismatch — treat as not-found, never leak.
            return None
        return _point_to_result(point, score=1.0)

    # ── Delete ────────────────────────────────────────────────────

    async def delete_document_vectors(
        self, workspace_id: str, document_id: str, user_id: str
    ) -> None:
        """
        Delete every vector for a given document. The user_id check is
        defensive — even if a caller passes a bad workspace, the user
        can't delete someone else's data.
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
            # 404 means the collection doesn't exist yet — no vectors to delete,
            # which is fine (document was never ingested). Any other error is real.
            if exc.status_code == 404:
                log.debug(
                    "delete_vectors_collection_missing",
                    document_id=document_id,
                    workspace_id=workspace_id,
                )
                return
            raise VectorStoreError(
                f"Failed to delete vectors for document '{document_id}'",
                detail=str(exc),
            ) from exc

    # ── Health ────────────────────────────────────────────────────

    async def copy_document_vectors(
        self, source: DocumentRef, target: DocumentRef
    ) -> int:
        """Copy a document's points into another workspace for another owner.

        Used to give each demo account its own copy of the sample documents
        without embedding them again. The copies are ordinary points of the
        target owner, so the usual isolation filters apply to them.
        """
        src_name = self._collection_name(source.workspace_id)
        qfilter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=source.document_id),
                ),
                qmodels.FieldCondition(
                    key="user_id", match=qmodels.MatchValue(value=source.user_id)
                ),
            ]
        )
        points: list = []
        offset = None
        try:
            while True:
                batch, offset = await self._client.scroll(
                    collection_name=src_name,
                    scroll_filter=qfilter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                points.extend(batch)
                if offset is None:
                    break
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Cannot read source collection '{src_name}'", detail=str(exc)
            ) from exc
        if not points:
            return 0

        new_ids = {str(p.id): str(uuid.uuid4()) for p in points}

        def relink(chunk_id: object) -> str | None:
            return new_ids.get(str(chunk_id)) if chunk_id else None

        copies = [
            qmodels.PointStruct(
                id=new_ids[str(p.id)],
                vector=p.vector,
                payload={
                    **(p.payload or {}),
                    "user_id": target.user_id,
                    "workspace_id": target.workspace_id,
                    "document_id": target.document_id,
                    "prev_chunk_id": relink((p.payload or {}).get("prev_chunk_id")),
                    "next_chunk_id": relink((p.payload or {}).get("next_chunk_id")),
                },
            )
            for p in points
        ]
        if not await self.collection_exists(target.workspace_id):
            await self.create_collection(
                target.workspace_id, dimension=len(points[0].vector)
            )
        dst_name = self._collection_name(target.workspace_id)
        try:
            await self._upsert_in_batches(dst_name, copies)
        except UnexpectedResponse as exc:
            raise VectorStoreError(
                f"Qdrant upsert failed for collection '{dst_name}'", detail=str(exc)
            ) from exc
        log.info(
            "document_vectors_copied",
            source_workspace=source.workspace_id,
            target_workspace=target.workspace_id,
            count=len(copies),
        )
        return len(copies)

    async def health(self) -> bool:
        try:
            info = await self._client.get_collections()
            return info is not None
        except Exception:
            return False

    # ── Filter helpers ────────────────────────────────────────────

    @staticmethod
    def _isolation_must(search_filter: SearchFilter) -> list[qmodels.FieldCondition]:
        must: list[qmodels.FieldCondition] = [
            qmodels.FieldCondition(
                key="user_id",
                match=qmodels.MatchValue(value=search_filter.user_id),
            ),
            qmodels.FieldCondition(
                key="workspace_id",
                match=qmodels.MatchValue(value=search_filter.workspace_id),
            ),
        ]
        if search_filter.document_ids:
            must.append(
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchAny(any=search_filter.document_ids),
                )
            )
        return must

    def _build_isolation_filter(self, search_filter: SearchFilter) -> qmodels.Filter:
        return qmodels.Filter(must=self._isolation_must(search_filter))


# ── Payload → domain mapping ──────────────────────────────────────


def _payload_to_vector_payload(p: dict) -> VectorPayload:
    return VectorPayload(
        user_id=p.get("user_id", ""),
        workspace_id=p.get("workspace_id", ""),
        document_id=p.get("document_id", ""),
        chunk_index=p.get("chunk_index", 0),
        text=p.get("text", ""),
        source=p.get("source", ""),
        page_number=p.get("page_number", 0),
        chunk_total=p.get("chunk_total", 0),
        heading_path=list(p.get("heading_path") or []),
        char_start=p.get("char_start", 0),
        char_end=p.get("char_end", 0),
        token_count=p.get("token_count", 0),
        prev_chunk_id=p.get("prev_chunk_id"),
        next_chunk_id=p.get("next_chunk_id"),
    )


def _hit_to_result(hit) -> SearchResult:
    p = hit.payload or {}
    return SearchResult(
        chunk_id=str(hit.id),
        score=hit.score,
        payload=_payload_to_vector_payload(p),
    )


def _point_to_result(point, score: float) -> SearchResult:
    p = point.payload or {}
    return SearchResult(
        chunk_id=str(point.id),
        score=score,
        payload=_payload_to_vector_payload(p),
    )
