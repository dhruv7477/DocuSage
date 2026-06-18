"""
src/vector_store/base.py
─────────────────────────
Abstract interface for the vector store layer.

Key design decisions:
1. VectorPayload carries ALL metadata needed to reconstruct context
   without a second DB lookup in the hot path.
2. user_id is a REQUIRED field in both upsert and search — this is
   the structural enforcement of user-level data isolation.
3. SearchResult wraps the payload + score so the RAG service
   can apply threshold filtering without knowing the DB internals.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class VectorPayload:
    """
    Metadata stored alongside each vector in the DB.

    This is the complete context needed to:
    - Enforce isolation (user_id, workspace_id)
    - Reconstruct the chunk for prompt building (text, source, page,
      heading_path)
    - Render precise UI citations (char_start / char_end identify the
      exact passage in the original `full_text`)
    - Walk neighbouring chunks via prev_chunk_id / next_chunk_id when
      we need extra context for an answer
    """
    user_id: str           # ★ ISOLATION KEY — every search MUST filter on this
    workspace_id: str
    document_id: str
    chunk_index: int
    text: str              # The actual chunk text, stored for prompt building
    source: str            # Original filename
    page_number: int = 0
    chunk_total: int = 0   # Total chunks in the document (for context)
    heading_path: list[str] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None


@dataclass
class SearchResult:
    """A single result from a similarity search."""
    chunk_id: str
    score: float           # Cosine similarity 0.0–1.0
    payload: VectorPayload


@dataclass
class SearchFilter:
    """
    Explicit filter passed to every search.
    Having this as a typed object prevents accidentally omitting user_id.
    """
    user_id: str           # REQUIRED — never search without this
    workspace_id: str      # REQUIRED — tenant isolation
    document_ids: list[str] = field(default_factory=list)  # Optional: search within specific docs


class BaseVectorStore(ABC):
    """
    Abstract contract every vector store must satisfy.
    The application ONLY imports this class — never a concrete store.
    """

    @abstractmethod
    async def create_collection(self, workspace_id: str, dimension: int) -> None:
        """
        Create a vector collection for a workspace if it doesn't exist.
        Called once when a workspace is created.
        dimension must match the embedding provider's output dimension.
        """
        ...

    @abstractmethod
    async def delete_collection(self, workspace_id: str) -> None:
        """
        Delete the entire collection for a workspace.
        Called when a workspace is deleted.
        """
        ...

    @abstractmethod
    async def upsert_vectors(
        self,
        workspace_id: str,
        chunk_ids: list[str],
        vectors: list[list[float]],
        payloads: list[VectorPayload],
    ) -> None:
        """
        Insert or update vectors in the collection.
        chunk_ids, vectors, and payloads must be the same length.
        """
        ...

    @abstractmethod
    async def search(
        self,
        workspace_id: str,
        query_vector: list[float],
        search_filter: SearchFilter,
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[SearchResult]:
        """
        Perform similarity search with mandatory user+workspace isolation filter.

        Args:
            workspace_id:    Which collection to search
            query_vector:    Embedded query from the embedding provider
            search_filter:   MUST include user_id — enforces data isolation
            top_k:           Maximum results to return
            score_threshold: Minimum cosine similarity (0.0–1.0)

        Returns:
            List of SearchResult sorted by score descending.
        """
        ...

    @abstractmethod
    async def delete_document_vectors(
        self, workspace_id: str, document_id: str, user_id: str
    ) -> None:
        """
        Delete all vectors belonging to a specific document.
        Called when a document is deleted. Scoped to user_id for safety.
        """
        ...

    @abstractmethod
    async def collection_exists(self, workspace_id: str) -> bool:
        """Returns True if the collection for this workspace exists."""
        ...

    async def text_search(
        self,
        workspace_id: str,
        query: str,
        search_filter: "SearchFilter",
        top_k: int = 50,
    ) -> list["SearchResult"]:
        """
        Sparse / keyword retrieval over the same isolation filter as
        `search()`. Default raises NotImplementedError so legacy stores
        keep working; backends that support a full-text payload index
        (Qdrant) override this. The Phase 3 retrieval pipeline calls it
        in parallel with vector search to produce real hybrid results.
        """
        raise NotImplementedError("This vector store does not support text search.")

    async def list_document_chunks(
        self,
        workspace_id: str,
        document_id: str,
        user_id: str,
    ) -> list["SearchResult"]:
        """
        Return every chunk of one document (ordered by chunk_index),
        within the user's isolation filter. Powers the doc-summary
        route, which needs the full document text rather than a search.
        Default raises so legacy stores keep working.
        """
        raise NotImplementedError("This vector store does not support chunk listing.")

    async def get_chunk(
        self,
        workspace_id: str,
        chunk_id: str,
        user_id: str,
    ) -> "SearchResult | None":
        """
        Fetch a single chunk by its point id. Returns None when the
        chunk doesn't exist or belongs to another user (ownership is
        enforced at the store layer — defence in depth).
        """
        raise NotImplementedError("This vector store does not support chunk lookup.")

    @abstractmethod
    async def health(self) -> bool:
        """Returns True if the vector store is reachable."""
        ...

    async def get_collection_dimension(self, workspace_id: str) -> int | None:
        """
        Return the configured vector dimension for the workspace's
        collection, or None if the collection does not exist. Used by
        `upsert_vectors` to validate the embedder's output dimension
        matches what the collection was created with.
        """
        return None