from pathlib import Path

from FlagEmbedding import BGEM3FlagModel, FlagReranker

cache_dir = Path(__file__).resolve().parent.parent / "models"
cache_dir.mkdir(parents=True, exist_ok=True)

print("Downloading BGE-M3 embedding model (~2.2GB)...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, cache_dir=str(cache_dir))
print("BGE-M3 downloaded.")

print("Downloading BGE reranker (~1.1GB)...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, cache_dir=str(cache_dir))
print("Reranker downloaded.")

print("All models cached successfully.")
