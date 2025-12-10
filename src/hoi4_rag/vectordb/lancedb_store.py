"""LanceDB vector store integration for HOI4 RAG."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector
from pydantic import ConfigDict, Field


class IndexError(Exception):
    """Raised when vector index operations fail."""


class DocumentChunk(LanceModel):  # type: ignore[misc]
    """Schema for chunk storage in LanceDB."""

    id: str = Field(json_schema_extra={"nullable": False})  # primary key
    text: str = Field(json_schema_extra={"nullable": False})
    dense_vector: Vector(1024)  # type: ignore[valid-type]
    file_path: str = Field(json_schema_extra={"nullable": False})
    domain: str = Field(json_schema_extra={"nullable": False})
    concept: str = Field(json_schema_extra={"nullable": False})
    version: str = Field(json_schema_extra={"nullable": False})
    section_heading: str = Field(json_schema_extra={"nullable": False})
    section_anchor: str = Field(json_schema_extra={"nullable": False})
    parent_headings: list[str] = Field(json_schema_extra={"nullable": False})
    token_count: int = Field(json_schema_extra={"nullable": False})
    content_hash: str = Field(json_schema_extra={"nullable": False})
    ingestion_timestamp: datetime = Field(
        json_schema_extra={"nullable": False},
        default_factory=lambda: datetime.now(timezone(timedelta(hours=1), "CET")),
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, primary_key="id")  # type: ignore


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str]


class LanceDBStore:
    """Wrapper around LanceDB for dense vector storage."""

    def __init__(self, db_path: Path, table_name: str = "hoi4_docs") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.LanceTable | None = None

    def initialize(self) -> None:
        """Create database and table if they do not exist."""
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))

        if self.table_name in self._db.table_names():
            self._table = self._db.open_table(self.table_name)
            return

        self._table = self._db.create_table(self.table_name, schema=DocumentChunk)

    def _require_table(self) -> lancedb.table.LanceTable:
        if self._table is None or self._db is None:
            self.initialize()
        if self._table is None:
            raise IndexError("Failed to initialize LanceDB table")
        return self._table

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Insert or update chunks. Returns number of modified chunks."""
        if not chunks:
            return 0

        table = self._require_table()

        df = table.to_pandas() if len(table) > 0 else None

        existing_hashes = (
            {row["id"]: row["content_hash"] for _, row in df[["id", "content_hash"]].iterrows()}
            if df is not None and not df.empty
            else {}
        )

        payload = []
        for chunk in chunks:
            current_hash = existing_hashes.get(chunk.id)
            if current_hash is not None and current_hash == chunk.content_hash:
                continue
            payload.append(chunk.model_dump())

        if not payload:
            return 0

        payload_table = pa.Table.from_pylist(payload, schema=table.schema)

        res = (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(payload_table)
        )

        return int(res.num_inserted_rows + res.num_updated_rows)

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        """Delete chunks by id. Returns number deleted."""
        if not chunk_ids:
            return 0

        table = self._require_table()
        ids_list = ",".join(f"'{cid}'" for cid in chunk_ids)
        table.delete(f"id IN ({ids_list})")
        return int(len(ids_list.split(",")))

    def search_dense(
        self, query_vector: np.ndarray, top_k: int, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        """Dense vector similarity search with optional metadata filters."""
        table = self._require_table()
        filter_expr = None
        if filters:
            clauses = [f"{key} == '{value}'" for key, value in filters.items()]
            filter_expr = " AND ".join(clauses)

        results = (
            table.search(query_vector, vector_column_name="dense_vector")
            .where(filter_expr)
            .limit(top_k)
            .to_pandas()
        )

        search_results: list[SearchResult] = []
        for _, row in results.iterrows():
            metadata = {
                "file_path": row["file_path"],
                "domain": row["domain"],
                "concept": row["concept"],
                "version": row["version"],
                "section_heading": row["section_heading"],
                "section_anchor": row["section_anchor"],
            }
            score = float(row["_distance"]) if "_distance" in row else float(row.get("score", 0))
            search_results.append(
                SearchResult(chunk_id=row["id"], text=row["text"], score=score, metadata=metadata)
            )
        return search_results

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a chunk by id."""
        table = self._require_table()
        df = table.to_pandas()
        row = df.loc[df["id"] == chunk_id]
        if row.empty:
            return None
        return DocumentChunk(**row.iloc[0].to_dict())

    def get_all_chunk_ids(self) -> set[str]:
        """Return all chunk ids present in the table."""
        table = self._require_table()
        if len(table) == 0:
            return set()
        df = table.to_pandas()
        return set(df["id"].tolist())

    def create_fts_index(self) -> None:
        """Create a full-text search index on text field."""
        table = self._require_table()
        try:
            table.create_fts_index("text")
        except Exception as exc:  # pragma: no cover - defensive
            raise IndexError("Failed to create FTS index") from exc
