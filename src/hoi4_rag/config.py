"""Configuration management for HOI4 RAG system.

This module provides centralized configuration using pydantic-settings with
environment variable support.
"""

from pathlib import Path
from typing import ClassVar

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseSettings):
    """Configuration for BGE-M3 embedding model."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        case_sensitive=False,
    )

    model_name: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model name for embeddings",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 precision for GPU inference",
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Compute device (cuda/cpu/mps)",
    )
    max_length: int = Field(
        default=8192,
        description="Maximum sequence length (BGE-M3 context limit)",
    )
    batch_size: int = Field(
        default=8,
        description="Batch size for embedding generation",
    )


class RerankerConfig(BaseSettings):
    """Configuration for BGE reranker model."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="RERANKER_",
        case_sensitive=False,
    )

    model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace model name for reranking",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 precision for GPU inference",
    )
    top_k: int = Field(
        default=10,
        description="Number of results to return after reranking",
    )


class RetrievalConfig(BaseSettings):
    """Configuration for hybrid search and retrieval."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        case_sensitive=False,
    )

    top_k_dense: int = Field(
        default=50,
        description="Number of results from dense search",
    )
    top_k_sparse: int = Field(
        default=50,
        description="Number of results from sparse search",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant for rank fusion",
    )
    dense_weight: float = Field(
        default=0.4,
        description="Weight for dense retrieval in fusion",
    )
    sparse_weight: float = Field(
        default=0.2,
        description="Weight for sparse retrieval in fusion",
    )
    colbert_weight: float = Field(
        default=0.4,
        description="Weight for ColBERT reranking in fusion",
    )


class PathConfig(BaseSettings):
    """Configuration for file paths."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
    )

    knowledge_base_path: Path = Field(
        default=Path("./data/raw"),
        description="Path to knowledge base markdown files",
    )
    vectordb_path: Path = Field(
        default=Path("./vectordb"),
        description="Path to LanceDB vector database",
    )
    model_cache_path: Path = Field(
        default=Path("./models"),
        description="Path to cache downloaded models",
    )


class Settings(BaseSettings):
    """Main settings class with all configuration."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Nested configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    # MCP Server settings
    mcp_server_name: str = Field(
        default="hoi4-modding-rag",
        description="Name of the MCP server",
    )
    mcp_log_level: str = Field(
        default="INFO",
        description="Logging level for MCP server",
    )

    # Development settings
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables and .env file.

        Returns:
            Settings: Configured settings instance
        """
        return cls()

    def validate_paths(self) -> None:
        """Ensure required directories exist."""
        self.paths.vectordb_path.mkdir(parents=True, exist_ok=True)
        self.paths.model_cache_path.mkdir(parents=True, exist_ok=True)

        if not self.paths.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base not found at {self.paths.knowledge_base_path}")
