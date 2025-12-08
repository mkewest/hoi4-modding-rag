from FlagEmbedding import BGEM3FlagModel, FlagReranker

print("Downloading BGE-M3 embedding model (~2.2GB)...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("BGE-M3 downloaded.")

print("Downloading BGE reranker (~1.1GB)...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
print("Reranker downloaded.")

print("All models cached successfully.")
