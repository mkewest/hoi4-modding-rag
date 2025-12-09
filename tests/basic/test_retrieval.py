import numpy as np
from hoi4_rag.config import RetrievalConfig
from hoi4_rag.embeddings import QueryEmbedding
from hoi4_rag.retrieval import HybridSearcher
from hoi4_rag.vectordb import SearchResult


class FakeEmbedder:
    def embed_query(self, text: str) -> QueryEmbedding:
        del text
        return QueryEmbedding(
            dense=np.ones((1, 1024), dtype=np.float32),
            sparse={1: 0.5},
            colbert=np.ones((2, 1024), dtype=np.float32),
        )


class FakeStore:
    def __init__(self):
        self._data = {
            "a#1": SearchResult(
                chunk_id="a#1",
                text="alpha",
                score=0.0,
                metadata={
                    "domain": "core",
                    "concept": "c",
                    "version": "1",
                    "section_heading": "h",
                    "section_anchor": "a",
                    "file_path": "a",
                },
            )
        }

    def search_dense(self, query_vector, top_k, filters=None):
        del query_vector, top_k, filters
        return [self._data["a#1"]]

    def get_chunk_by_id(self, cid):
        sr = self._data.get(cid)
        if sr:
            return type("Obj", (), {**sr.metadata, "id": sr.chunk_id, "text": sr.text})
        return None


class FakeSparse:
    def search(self, query_sparse, top_k):
        del query_sparse, top_k
        return [("a#1", 0.3)]


def test_hybrid_search_rrf_and_rerank():
    searcher = HybridSearcher(
        embedder=FakeEmbedder(),
        lancedb_store=FakeStore(),
        sparse_index=FakeSparse(),
        config=RetrievalConfig(),
    )
    results = searcher.search("query", top_k=1, use_colbert_rerank=False)
    assert results
    assert results[0].chunk_id == "a#1"
