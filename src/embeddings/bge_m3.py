"""BGEM3 embedding wrapper for HOI4 RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from config import EmbeddingConfig
from FlagEmbedding import BGEM3FlagModel


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


@dataclass
class EmbeddingOutput:
    dense: np.ndarray
    sparse: list[dict[int, float]]
    colbert: list[np.ndarray]


@dataclass
class QueryEmbedding:
    dense: np.ndarray
    sparse: dict[int, float]
    colbert: np.ndarray


class BGEM3Embedder:
    """Wrapper around FlagEmbedding BGEM3 model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model: BGEM3FlagModel | None = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            self._model = BGEM3FlagModel(
                self.config.model_name,
                use_fp16=self.config.use_fp16,
                device=self.config.device,
            )
            self._tokenizer = self._model.tokenizer
        except Exception as exc:  # pragma: no cover - defensive
            raise EmbeddingError(f"Failed to load model {self.config.model_name}") from exc

    @staticmethod
    def _pick(
        outputs: dict[str, Any],
        primary: str,
        fallbacks: tuple[str, ...],
        default: list[Any] | None = None,
    ) -> list[Any]:
        """Return outputs[primary] or first available fallback as a list (converting ndarrays)."""

        def _as_list(value: object) -> list[Any]:
            if isinstance(value, np.ndarray):
                return cast(list[Any], value.tolist())
            if isinstance(value, list | tuple):
                return list(value)
            # Unknown type: wrap as single-element list
            return [value]

        if primary in outputs:
            return _as_list(outputs[primary])
        for key in fallbacks:
            if key in outputs:
                return _as_list(outputs[key])
        return default if default is not None else []

    def embed_documents(self, texts: list[str]) -> EmbeddingOutput:
        """Generate embeddings for multiple documents."""
        self._ensure_model()
        assert self._model is not None

        if not texts:
            empty_dense = np.empty((0, 1024), dtype=np.float32)
            return EmbeddingOutput(dense=empty_dense, sparse=[], colbert=[])

        dense_chunks: list[np.ndarray] = []
        sparse_vectors: list[dict[int, float]] = []
        colbert_vectors: list[np.ndarray] = []

        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                outputs = self._model.encode(
                    batch,
                    batch_size=len(batch),
                    max_length=self.config.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True,
                )
            except Exception as exc:  # pragma: no cover - defensive
                raise EmbeddingError("Failed to generate embeddings") from exc

            dense_raw = self._pick(outputs, "dense_vecs", ("dense_embeddings", "dense"), [])
            dense_batch = np.asarray(dense_raw, dtype=np.float32)
            dense_chunks.append(dense_batch)

            sparse_vectors.extend(
                self._pick(outputs, "sparse_vecs", ("lexical_weights", "sparse"), [])
            )

            colbert_batch = [
                np.asarray(vec, dtype=np.float32)
                for vec in self._pick(
                    outputs, "colbert_vecs", ("colbert_embeddings", "colbert"), []
                )
            ]
            colbert_vectors.extend(colbert_batch)

            if self.config.device.startswith("cuda"):
                torch.cuda.empty_cache()

        dense = np.vstack(dense_chunks) if dense_chunks else np.empty((0, 1024), dtype=np.float32)
        return EmbeddingOutput(dense=dense, sparse=sparse_vectors, colbert=colbert_vectors)

    def embed_query(self, text: str) -> QueryEmbedding:
        """Generate embeddings for a single query."""
        self._ensure_model()
        assert self._model is not None

        try:
            outputs = self._model.encode(
                [text],
                batch_size=1,
                max_length=self.config.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise EmbeddingError("Failed to generate query embedding") from exc

        dense_raw = self._pick(outputs, "dense_vecs", ("dense_embeddings", "dense"), [])
        if len(dense_raw) == 0:
            raise EmbeddingError("Model returned no dense embeddings")

        sparse_raw = self._pick(outputs, "sparse_vecs", ("lexical_weights", "sparse"), [{}])
        colbert_raw = self._pick(outputs, "colbert_vecs", ("colbert_embeddings", "colbert"), [[]])

        return QueryEmbedding(
            dense=np.asarray(dense_raw[0], dtype=np.float32),
            sparse=sparse_raw[0] if sparse_raw and len(sparse_raw) > 0 else {},
            colbert=(
                np.asarray(colbert_raw[0], dtype=np.float32)
                if colbert_raw and len(colbert_raw) > 0
                else np.empty((0, 0))
            ),
        )

    def token_id_to_string(self, token_id: int) -> str:
        """Convert token id to text token."""
        self._ensure_model()
        assert self._model is not None
        assert self._tokenizer is not None

        token_list = self._tokenizer.convert_ids_to_tokens([token_id])
        return token_list[0]

    @staticmethod
    def compute_colbert_score(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
        """Compute ColBERT MaxSim score."""
        if query_vecs.size == 0 or doc_vecs.size == 0:
            return 0.0

        similarity = np.matmul(query_vecs, doc_vecs.T)
        max_per_query = similarity.max(axis=1)
        return float(max_per_query.mean())
