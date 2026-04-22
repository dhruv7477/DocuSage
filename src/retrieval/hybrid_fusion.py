"""
src/retrieval/hybrid_fusion.py
───────────────────────────────
Reciprocal Rank Fusion (RRF) for hybrid search result merging.

Why RRF and not score normalisation?
    Different retrieval systems (dense vector, sparse BM25) produce
    scores on completely different scales.  Normalising and adding
    raw scores is fragile — a BM25 score of 0.9 is not comparable
    to a cosine similarity of 0.9.

    RRF instead uses RANK position (1st, 2nd, 3rd...) which is
    scale-independent.  It's robust and works well in practice.

Algorithm:
    For each result in each result list, its RRF contribution is:
        1 / (k + rank)
    where k=60 is a smoothing constant (standard literature default).

    Results from multiple lists are fused by summing their RRF scores
    across all lists.  Results only in one list get the other list's
    contribution as 0 (they simply don't appear in it).

Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual rank learning methods"
"""

from __future__ import annotations

from dataclasses import dataclass

from src.vector_store.base import SearchResult


# RRF smoothing constant — 60 is the standard value from the literature
_RRF_K = 60


@dataclass
class FusedResult:
    """A search result with its fused RRF score and provenance."""
    result: SearchResult
    rrf_score: float
    from_vector: bool = False
    from_bm25: bool = False

    @property
    def chunk_id(self) -> str:
        return self.result.chunk_id


def reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int = _RRF_K,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Merge vector and BM25 results using Reciprocal Rank Fusion.

    Args:
        vector_results:  Dense retrieval results (sorted by cosine score desc).
        bm25_results:    Sparse BM25 results (sorted by BM25 score desc).
        k:               RRF smoothing constant (default 60).
        top_k:           Number of fused results to return.

    Returns:
        Merged SearchResult list sorted by RRF score descending.
        The .score field on each result is the normalised RRF score [0, 1].
    """
    scores: dict[str, float] = {}
    # Map chunk_id → SearchResult for merging payloads
    result_map: dict[str, SearchResult] = {}
    provenance: dict[str, dict] = {}

    # Score vector results
    for rank, result in enumerate(vector_results, start=1):
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        result_map[cid] = result
        provenance.setdefault(cid, {})["from_vector"] = True

    # Score BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        cid = result.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        # Prefer the vector result's payload if available (has original score)
        if cid not in result_map:
            result_map[cid] = result
        provenance.setdefault(cid, {})["from_bm25"] = True

    if not scores:
        return []

    # Normalise RRF scores to [0, 1]
    max_rrf = max(scores.values())
    normalised = {cid: s / max_rrf for cid, s in scores.items()}

    # Sort by RRF score descending
    sorted_ids = sorted(normalised, key=lambda c: normalised[c], reverse=True)

    fused: list[SearchResult] = []
    for cid in sorted_ids[:top_k]:
        original = result_map[cid]
        # Replace the score with the normalised RRF score
        fused.append(SearchResult(
            chunk_id=cid,
            score=round(normalised[cid], 4),
            payload=original.payload,
        ))

    return fused


def deduplicate_results(
    results: list[SearchResult],
    similarity_threshold: float = 0.85,
) -> list[SearchResult]:
    """
    Remove near-duplicate chunks from the result list.

    Two chunks are considered duplicates if their text overlap exceeds
    `similarity_threshold` using a simple Jaccard similarity on word sets.

    Why this matters:
        When multi-query rewriting is used, the same chunk can appear
        multiple times (via different queries) with slightly different
        scores.  Keeping duplicates wastes the LLM context window.

    Args:
        results:              Sorted search results (best first).
        similarity_threshold: Jaccard threshold above which to drop.

    Returns:
        Deduplicated list preserving the best (first) occurrence.
    """
    seen_texts: list[set[str]] = []
    deduped: list[SearchResult] = []

    for result in results:
        words = set(result.payload.text.lower().split())
        is_dup = False
        for seen in seen_texts:
            if not words or not seen:
                continue
            jaccard = len(words & seen) / len(words | seen)
            if jaccard >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            deduped.append(result)
            seen_texts.append(words)

    return deduped
