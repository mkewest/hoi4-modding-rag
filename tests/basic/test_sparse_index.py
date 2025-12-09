from pathlib import Path

from hoi4_rag.vectordb import SparseIndex


def test_sparse_index_add_search(tmp_path: Path):
    index = SparseIndex(tmp_path)
    index.add_document("doc1", {1: 0.5, 2: 0.2})
    index.add_document("doc2", {1: 0.1, 3: 0.9})
    index.compute_idf_scores()

    results = index.search({1: 0.4, 3: 0.6}, top_k=2)
    assert results[0][0] in {"doc1", "doc2"}
    assert len(results) == 2


def test_sparse_index_save_load(tmp_path: Path):
    index = SparseIndex(tmp_path)
    index.add_document("doc1", {1: 1.0})
    index.compute_idf_scores()
    index.save()

    reloaded = SparseIndex(tmp_path)
    assert "doc1" in reloaded.doc_to_tokens
