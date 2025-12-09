from pathlib import Path

import numpy as np
from hoi4_rag.vectordb import DocumentChunk, LanceDBStore


def test_lancedb_upsert_and_search(tmp_path: Path):
    store = LanceDBStore(tmp_path)
    store.initialize()

    chunk = DocumentChunk(
        id="doc#sec",
        text="some text",
        dense_vector=np.ones(1024).tolist(),
        file_path="doc",
        domain="core",
        concept="c",
        version="1",
        section_heading="H",
        section_anchor="sec",
        parent_headings=["H"],
        token_count=10,
        content_hash="hash",
        ingestion_timestamp=None,
    )

    count = store.upsert_chunks([chunk])
    assert count == 1

    results = store.search_dense(np.ones(1024), top_k=1, filters={"domain": "core"})
    assert results
    assert results[0].chunk_id == "doc#sec"

    deleted = store.delete_chunks(["doc#sec"])
    assert deleted == 1
