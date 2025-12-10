"""Custom sparse inverted index for BGE-M3 sparse vectors."""

from __future__ import annotations

import json
import math
from pathlib import Path


class IndexError(Exception):
    """Raised when sparse index operations fail."""


class SparseIndex:
    """Inverted index for sparse vectors."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = Path(index_path)
        self.index_file = self.index_path / "sparse_index.json"

        self.token_to_docs: dict[int, dict[str, float]] = {}
        self.doc_to_tokens: dict[str, dict[int, float]] = {}
        self.idf_scores: dict[int, float] = {}
        self.total_docs: int = 0

        if self.index_file.exists():
            self.load()

    def add_document(self, doc_id: str, sparse_vector: dict[int, float]) -> None:
        """Index a single document."""
        if doc_id in self.doc_to_tokens:
            self.remove_document(doc_id)

        self.doc_to_tokens[doc_id] = dict(sparse_vector)
        for token_id, weight in sparse_vector.items():
            token_docs = self.token_to_docs.setdefault(int(token_id), {})
            token_docs[doc_id] = float(weight)

        self.total_docs = len(self.doc_to_tokens)

    def add_documents(self, doc_sparse_pairs: list[tuple[str, dict[int, float]]]) -> None:
        """Batch index documents."""
        for doc_id, sparse_vector in doc_sparse_pairs:
            self.add_document(doc_id, sparse_vector)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from the index."""
        tokens = self.doc_to_tokens.pop(doc_id, None)
        if tokens is None:
            return

        for token_id in list(tokens.keys()):
            docs = self.token_to_docs.get(token_id, {})
            docs.pop(doc_id, None)
            if not docs:
                self.token_to_docs.pop(token_id, None)

        self.total_docs = len(self.doc_to_tokens)

    def search(self, query_sparse: dict[int, float], top_k: int) -> list[tuple[str, float]]:
        """Return top_k documents by sparse similarity."""
        scores: dict[str, float] = {}

        for token_id, q_weight in query_sparse.items():
            docs = self.token_to_docs.get(int(token_id))
            if not docs:
                continue

            idf = self.idf_scores.get(int(token_id), 1.0)
            for doc_id, d_weight in docs.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + q_weight * d_weight * idf

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def compute_idf_scores(self) -> None:
        """Compute IDF scores for all tokens."""
        if self.total_docs == 0:
            self.idf_scores = {}
            return

        idf: dict[int, float] = {}
        for token_id, docs in self.token_to_docs.items():
            df = len(docs)
            idf[token_id] = math.log((self.total_docs + 1) / (df + 1)) + 1
        self.idf_scores = idf

    def save(self) -> None:
        """Persist index to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_docs": {str(k): v for k, v in self.token_to_docs.items()},
            "doc_to_tokens": {
                doc: {str(k): v for k, v in tokens.items()}
                for doc, tokens in self.doc_to_tokens.items()
            },
            "idf_scores": {str(k): v for k, v in self.idf_scores.items()},
            "total_docs": self.total_docs,
        }
        with self.index_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self) -> None:
        """Load index from disk."""
        try:
            with self.index_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            raise IndexError("Failed to load sparse index") from exc

        self.token_to_docs = {
            int(k): {doc: float(w) for doc, w in v.items()}
            for k, v in payload.get("token_to_docs", {}).items()
        }
        self.doc_to_tokens = {
            doc: {int(k): float(w) for k, w in tokens.items()}
            for doc, tokens in payload.get("doc_to_tokens", {}).items()
        }
        self.idf_scores = {int(k): float(v) for k, v in payload.get("idf_scores", {}).items()}
        self.total_docs = int(payload.get("total_docs", len(self.doc_to_tokens)))

    def clear(self) -> None:
        """Reset the index and remove persisted file."""
        self.token_to_docs.clear()
        self.doc_to_tokens.clear()
        self.idf_scores.clear()
        self.total_docs = 0
        if self.index_file.exists():
            self.index_file.unlink()
