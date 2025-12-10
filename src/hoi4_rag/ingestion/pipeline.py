"""Ingestion pipeline for HOI4 RAG."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from hoi4_rag.chunking import Chunk, MarkdownChunker
from hoi4_rag.config import Settings
from hoi4_rag.embeddings import BGEM3Embedder
from hoi4_rag.vectordb import DocumentChunk, LanceDBStore, SparseIndex


class IngestionError(Exception):
    """Raised when ingestion fails."""


@dataclass
class IngestionStats:
    documents_processed: int
    chunks_created: int
    chunks_updated: int
    chunks_deleted: int
    embedding_time_seconds: float
    indexing_time_seconds: float
    total_time_seconds: float


class IngestionPipeline:
    """Orchestrates markdown → chunks → embeddings → vector stores."""

    def __init__(
        self,
        chunker: MarkdownChunker,
        embedder: BGEM3Embedder,
        lancedb_store: LanceDBStore,
        sparse_index: SparseIndex,
        config: Settings,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.lancedb_store = lancedb_store
        self.sparse_index = sparse_index
        self.config = config

    def ingest_full(self, source_path: Path, force: bool = False) -> IngestionStats:
        """Full ingestion of the entire knowledge base."""
        start = datetime.now(timezone(timedelta(hours=1), "CET"))
        self.config.validate_paths()
        self.lancedb_store.initialize()

        chunks = self.chunker.chunk_directory(source_path)
        new_ids = {chunk.id for chunk in chunks}

        existing_hashes = self._existing_hashes()
        existing_ids = set(existing_hashes.keys())

        deleted_ids = existing_ids - new_ids
        chunks_to_process = (
            chunks if force else [c for c in chunks if existing_hashes.get(c.id) != c.content_hash]
        )

        embedding_start = datetime.now(timezone(timedelta(hours=1), "CET"))
        dense_vecs, sparse_vecs, colbert_vecs = self._embed_chunks(chunks_to_process)
        embedding_end = datetime.now(timezone(timedelta(hours=1), "CET"))

        # indexed_count = self._upsert(chunks_to_process, dense_vecs, colbert_vecs)

        self._update_sparse_index(chunks_to_process, sparse_vecs)

        deleted_count = self._delete(list(deleted_ids))

        indexing_end = datetime.now(timezone(timedelta(hours=1), "CET"))

        # Differentiate new vs updated chunks for clearer stats
        created_count = sum(1 for c in chunks_to_process if c.id not in existing_hashes)
        updated_count = len(chunks_to_process) - created_count

        return IngestionStats(
            documents_processed=len({Path(c.file_path) for c in chunks}),
            chunks_created=created_count,
            chunks_updated=updated_count,
            chunks_deleted=deleted_count,
            embedding_time_seconds=(embedding_end - embedding_start).total_seconds(),
            indexing_time_seconds=(indexing_end - embedding_end).total_seconds(),
            total_time_seconds=(indexing_end - start).total_seconds(),
        )

    def ingest_incremental(self, source_path: Path) -> IngestionStats:
        """Ingest only changed files."""
        return self.ingest_full(source_path, force=False)

    # Internal helpers
    def _existing_hashes(self) -> dict[str, str]:
        table = self.lancedb_store._require_table()
        if len(table) == 0:
            return {}
        df = table.to_pandas(columns=["id", "content_hash"])
        return {row["id"]: row["content_hash"] for _, row in df.iterrows()}

    def _embed_chunks(
        self, chunks: Sequence[Chunk]
    ) -> tuple[np.ndarray, list[dict[int, float]], list[np.ndarray]]:
        if not chunks:
            return np.empty((0, 1024), dtype=np.float32), [], []

        texts = [c.text for c in chunks]
        outputs = self.embedder.embed_documents(texts)
        return outputs.dense, outputs.sparse, outputs.colbert

    def _upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vecs: np.ndarray,
        colbert_vecs: Sequence[np.ndarray],
    ) -> int:
        if not chunks:
            return 0

        now = datetime.now(timezone(timedelta(hours=1), "CET"))
        document_chunks: list[DocumentChunk] = []
        colbert_store: dict[str, np.ndarray] = self._load_colbert_store()

        for chunk, dense, colbert in zip(chunks, dense_vecs, colbert_vecs, strict=False):
            doc = DocumentChunk(
                id=chunk.id,
                text=chunk.text,
                dense_vector=dense.tolist(),
                file_path=chunk.file_path,
                domain=chunk.domain,
                concept=chunk.concept,
                version=chunk.version,
                section_heading=chunk.section_heading,
                section_anchor=chunk.section_anchor,
                parent_headings=chunk.parent_headings,
                token_count=chunk.token_count,
                content_hash=chunk.content_hash,
                ingestion_timestamp=now,
            )
            document_chunks.append(doc)
            colbert_store[chunk.id] = colbert

        modified = self.lancedb_store.upsert_chunks(document_chunks)
        self._save_colbert_store(colbert_store)
        return modified

    def _delete(self, chunk_ids: Sequence[str]) -> int:
        if not chunk_ids:
            return 0
        deleted = self.lancedb_store.delete_chunks(list(chunk_ids))
        for cid in chunk_ids:
            self.sparse_index.remove_document(cid)
        self._prune_colbert_store(chunk_ids)
        self.sparse_index.save()
        return deleted

    def _update_sparse_index(
        self, chunks: Sequence[Chunk], sparse_vecs: Sequence[dict[int, float]]
    ) -> None:
        if not chunks:
            return
        pairs = [(chunk.id, sparse) for chunk, sparse in zip(chunks, sparse_vecs, strict=False)]
        self.sparse_index.add_documents(pairs)
        self.sparse_index.compute_idf_scores()
        self.sparse_index.save()

    def _load_colbert_store(self) -> dict[str, np.ndarray]:
        colbert_path = Path(self.config.paths.vectordb_path) / "colbert_vectors.npz"
        if not colbert_path.exists():
            return {}
        try:
            data = np.load(colbert_path, allow_pickle=True)
            return {key: data[key] for key in data.files}
        except Exception:
            return {}

    def _save_colbert_store(self, store: dict[str, np.ndarray]) -> None:
        colbert_path = Path(self.config.paths.vectordb_path) / "colbert_vectors.npz"
        if not store:
            return
        np.savez(colbert_path, **store)

    def _prune_colbert_store(self, removed_ids: Sequence[str]) -> None:
        if not removed_ids:
            return
        store = self._load_colbert_store()
        if not store:
            return
        for cid in removed_ids:
            store.pop(cid, None)
        self._save_colbert_store(store)
