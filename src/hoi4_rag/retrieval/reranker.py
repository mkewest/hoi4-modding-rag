"""Cross-encoder reranker using BGE reranker."""

from __future__ import annotations

from FlagEmbedding import FlagReranker

from hoi4_rag.config import RerankerConfig
from hoi4_rag.vectordb import SearchResult


class RetrievalError(Exception):
    """Raised when reranking fails."""


class BGEReranker:
    """Cross-encoder reranking with BGE reranker."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model: FlagReranker | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            self._model = FlagReranker(self.config.model_name, use_fp16=self.config.use_fp16)
        except Exception as exc:  # pragma: no cover - defensive
            raise RetrievalError(f"Failed to load reranker model {self.config.model_name}") from exc

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Rerank search results using cross-encoder."""
        if not candidates or top_k <= 0:
            return []

        self._ensure_model()
        pairs = [[query, c.text] for c in candidates]
        try:
            scores = self._model.compute_score(pairs, normalize=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise RetrievalError("Failed to rerank results") from exc

        scored = list(zip(candidates, scores, strict=False))
        reranked = sorted(scored, key=lambda item: item[1], reverse=True)
        top = reranked[:top_k]

        return [
            SearchResult(
                chunk_id=entry.chunk_id,
                text=entry.text,
                score=float(score),
                metadata=entry.metadata,
            )
            for entry, score in top
        ]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        self._ensure_model()
        try:
            score = self._model.compute_score([[query, document]], normalize=True)[0]
        except Exception as exc:  # pragma: no cover - defensive
            raise RetrievalError("Failed to score pair") from exc
        return float(score)
