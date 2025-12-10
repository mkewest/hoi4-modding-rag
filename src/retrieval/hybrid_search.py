"""Hybrid search combining dense, sparse, and optional ColBERT rerank."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from config import RetrievalConfig
from embeddings.bge_m3 import BGEM3Embedder
from vectordb import LanceDBStore, SearchResult, SparseIndex


class RetrievalError(Exception):
    """Raised when retrieval operations fail."""


class HybridSearcher:
    """Hybrid search with dense + sparse + optional ColBERT rerank."""

    def __init__(
        self,
        embedder: BGEM3Embedder,
        lancedb_store: LanceDBStore,
        sparse_index: SparseIndex,
        config: RetrievalConfig,
    ) -> None:
        self.embedder = embedder
        self.lancedb_store = lancedb_store
        self.sparse_index = sparse_index
        self.config = config

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
        use_colbert_rerank: bool = False,
    ) -> list[SearchResult]:
        """Main search interface."""
        if top_k <= 0:
            return []

        query_embedding = self.embedder.embed_query(query)

        dense_results = self.lancedb_store.search_dense(
            query_embedding.dense, self.config.top_k_dense, filters
        )

        sparse_pairs = self.sparse_index.search(query_embedding.sparse, self.config.top_k_sparse)
        sparse_results = self._hydrate_sparse_results(sparse_pairs)

        fused_ids = self._rrf_fusion(dense_results, sparse_results, k=self.config.rrf_k)
        fused_results = self._gather_results(fused_ids, dense_results, sparse_results)

        final_results = fused_results[:top_k]

        if use_colbert_rerank and final_results:
            final_results = self._colbert_rerank(
                query_embedding.colbert, final_results, top_k=top_k
            )

        return final_results

    def _hydrate_sparse_results(self, sparse_pairs: list[tuple[str, float]]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for doc_id, score in sparse_pairs:
            chunk = self.lancedb_store.get_chunk_by_id(doc_id)
            if chunk is None:
                continue
            metadata = {
                "file_path": chunk.file_path,
                "domain": chunk.domain,
                "concept": chunk.concept,
                "version": chunk.version,
                "section_heading": chunk.section_heading,
                "section_anchor": chunk.section_anchor,
            }
            results.append(
                SearchResult(
                    chunk_id=chunk.id, text=chunk.text, score=float(score), metadata=metadata
                )
            )
        return results

    def _gather_results(
        self,
        fused_ids: list[tuple[str, float]],
        dense_results: Sequence[SearchResult],
        sparse_results: Sequence[SearchResult],
    ) -> list[SearchResult]:
        dense_map = {r.chunk_id: r for r in dense_results}
        sparse_map = {r.chunk_id: r for r in sparse_results}

        gathered: list[SearchResult] = []
        for doc_id, score in fused_ids:
            src = dense_map.get(doc_id) or sparse_map.get(doc_id)
            if src is None:
                continue
            gathered.append(
                SearchResult(
                    chunk_id=src.chunk_id,
                    text=src.text,
                    score=float(score),
                    metadata=src.metadata,
                )
            )
        return gathered

    @staticmethod
    def _rrf_fusion(
        dense_results: Sequence[SearchResult],
        sparse_results: Sequence[SearchResult],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion of dense and sparse results."""
        scores: dict[str, float] = {}

        for rank, result in enumerate(dense_results):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank + 1)

        for rank, result in enumerate(sparse_results):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank + 1)

        fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return fused

    def _colbert_rerank(
        self, query_colbert: np.ndarray, candidates: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Rerank using ColBERT MaxSim if stored vectors exist."""
        colbert_store = Path(self.lancedb_store.db_path) / "colbert_vectors.npz"
        if not colbert_store.exists():
            return candidates[:top_k]

        try:
            store = np.load(colbert_store, allow_pickle=True)
        except Exception:
            return candidates[:top_k]

        scored: list[tuple[SearchResult, float]] = []
        for result in candidates[: top_k * 2]:
            if result.chunk_id not in store:
                continue
            doc_vecs = store[result.chunk_id]
            score = BGEM3Embedder.compute_colbert_score(query_colbert, doc_vecs)
            scored.append((result, score))

        if not scored:
            return candidates[:top_k]

        reranked = sorted(scored, key=lambda item: item[1], reverse=True)
        return [
            SearchResult(chunk_id=r.chunk_id, text=r.text, score=s, metadata=r.metadata)
            for r, s in reranked[:top_k]
        ]
