import numpy as np
from hoi4_rag.config import EmbeddingConfig
from hoi4_rag.embeddings import BGEM3Embedder, EmbeddingOutput


class DummyEmbedder(BGEM3Embedder):
    def _ensure_model(self) -> None:
        return

    def embed_documents(self, texts):
        dense = np.ones((len(texts), 1024), dtype=np.float32)
        sparse = [{0: 1.0} for _ in texts]
        colbert = [np.ones((2, 1024), dtype=np.float32) for _ in texts]
        return EmbeddingOutput(dense=dense, sparse=sparse, colbert=colbert)


def test_embed_documents_shapes():
    embedder = DummyEmbedder(EmbeddingConfig())
    out = embedder.embed_documents(["a", "b", "c"])
    assert out.dense.shape == (3, 1024)
    assert len(out.sparse) == 3
    assert len(out.colbert) == 3
    assert out.colbert[0].shape[1] == 1024


def test_batching_no_texts_returns_empty():
    embedder = DummyEmbedder(EmbeddingConfig())
    out = embedder.embed_documents([])
    assert out.dense.shape[0] == 0
    assert out.sparse == []
    assert out.colbert == []
