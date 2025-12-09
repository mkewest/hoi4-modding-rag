"""FastMCP server exposing HOI4 RAG tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hoi4_rag.config import Settings
from hoi4_rag.embeddings import BGEM3Embedder
from hoi4_rag.retrieval import BGEReranker, HybridSearcher
from hoi4_rag.vectordb import LanceDBStore, SparseIndex

try:  # Optional dependency guard for environments without MCP SDK
    from mcp import FastMCP, resource, tool
except ImportError:  # pragma: no cover - fallback stubs

    class FastMCP:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.tools = []
            self.resources = []

        def tool(self, *args: Any, **kwargs: Any):
            def decorator(func):
                self.tools.append(func)
                return func

            return decorator

        def resource(self, *args: Any, **kwargs: Any):
            def decorator(func):
                self.resources.append(func)
                return func

            return decorator

        def run(self) -> None:
            raise ImportError("mcp package not installed")

    def tool(*args: Any, **kwargs: Any):  # type: ignore
        def decorator(func):
            return func

        return decorator

    def resource(*args: Any, **kwargs: Any):  # type: ignore
        def decorator(func):
            return func

        return decorator


def _load_master_index(kb_path: Path) -> dict[str, Any]:
    """Load master_index.json if present."""
    master_index = kb_path / "master_index.json"
    if not master_index.exists():
        return {}
    try:
        with master_index.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _chunk_to_dict(chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "file_path": chunk.file_path,
        "domain": chunk.domain,
        "concept": chunk.concept,
        "version": chunk.version,
        "section_heading": chunk.section_heading,
        "section_anchor": chunk.section_anchor,
        "parent_headings": chunk.parent_headings,
    }


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Normalize search results to plain dicts."""
    # Prefer __dict__ for non-plain-dict objects (including dict subclasses)
    if hasattr(result, "__dict__") and not isinstance(result, dict):
        return dict(result.__dict__)
    if isinstance(result, dict):
        return dict(result)
    # Fallback for objects without __dict__
    keys = ("chunk_id", "text", "score", "metadata")
    return {key: getattr(result, key, None) for key in keys}


def create_server(
    settings: Settings | None = None,
    searcher: HybridSearcher | None = None,
    reranker: BGEReranker | None = None,
) -> FastMCP:
    """Build and return a configured FastMCP server."""
    settings = settings or Settings.from_env()

    # Lazy component assembly if not provided
    if searcher is None:
        embedder = BGEM3Embedder(settings.embedding)
        lancedb_store = LanceDBStore(settings.paths.vectordb_path)
        sparse_index = SparseIndex(settings.paths.vectordb_path)
        searcher = HybridSearcher(
            embedder=embedder,
            lancedb_store=lancedb_store,
            sparse_index=sparse_index,
            config=settings.retrieval,
        )
    if reranker is None:
        reranker = BGEReranker(settings.reranker)

    mcp = FastMCP(
        name=settings.mcp_server_name,
        description="HOI4 modding documentation retrieval system",
    )

    @mcp.tool()
    def search_documentation(query: str, domain: str | None = None, top_k: int = 5):
        """Search HOI4 modding documentation for technical information."""
        filters = {"domain": domain} if domain else None
        initial = searcher.search(query, top_k=top_k * 2, filters=filters, use_colbert_rerank=False)
        reranked = reranker.rerank(query, initial, top_k=top_k)
        return [_result_to_dict(r) for r in reranked]

    @mcp.tool()
    def search_code_examples(query: str, language: str = "pdx", top_k: int = 5):
        """Search for working code examples and implementation patterns."""
        del language  # not yet used for filtering
        initial = searcher.search(query, top_k=top_k * 2, use_colbert_rerank=False)
        reranked = reranker.rerank(query, initial, top_k=top_k)
        return [_result_to_dict(r) for r in reranked]

    @mcp.tool()
    def get_document_section(document_id: str, include_context: bool = False):
        """Retrieve a specific documentation section by ID."""
        chunk = searcher.lancedb_store.get_chunk_by_id(document_id)  # type: ignore[attr-defined]
        if chunk is None:
            return None
        if not include_context:
            return _chunk_to_dict(chunk)

        # Include siblings from same file
        table = searcher.lancedb_store._require_table()  # type: ignore[attr-defined]
        df = table.to_pandas(filter=f"file_path == '{chunk.file_path}'")
        siblings = []
        for _, row in df.iterrows():
            siblings.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "section_heading": row["section_heading"],
                    "section_anchor": row["section_anchor"],
                }
            )
        return {"section": _chunk_to_dict(chunk), "context": siblings}

    @mcp.tool()
    def lookup_define(define_name: str, category: str | None = None, top_k: int = 5):
        """Look up a specific HOI4 define value."""
        filters = {"domain": "defines_list"}
        query = define_name if category is None else f"{category} {define_name}"
        results = searcher.search(query, top_k=top_k, filters=filters, use_colbert_rerank=False)
        return [_result_to_dict(r) for r in results]

    @mcp.tool()
    def lookup_modifier(modifier_name: str, category: str | None = None, top_k: int = 5):
        """Look up a specific HOI4 modifier."""
        filters = {"domain": "modifiers_list"}
        query = modifier_name if category is None else f"{category} {modifier_name}"
        results = searcher.search(query, top_k=top_k, filters=filters, use_colbert_rerank=False)
        return [_result_to_dict(r) for r in results]

    @mcp.tool()
    def diagnose_error(error_message: str, context: str | None = None, top_k: int = 5):
        """Diagnose HOI4 modding errors from logs or descriptions."""
        combined = f"{error_message}\n{context}" if context else error_message
        results = searcher.search(combined, top_k=top_k, use_colbert_rerank=False)
        return [_result_to_dict(r) for r in results]

    @mcp.resource("hoi4://domains")
    def list_domains():
        """List available documentation domains."""
        index = _load_master_index(settings.paths.knowledge_base_path)
        domains = index.get("domains", {})
        return [{"name": name, "path": meta.get("path", "")} for name, meta in domains.items()]

    @mcp.resource("hoi4://domain/{domain_name}")
    def domain_overview(domain_name: str):
        """Get overview of a specific domain."""
        index = _load_master_index(settings.paths.knowledge_base_path)
        domains = index.get("domains", {})
        meta = domains.get(domain_name)
        if not meta:
            return None
        overview = (
            Path(settings.paths.knowledge_base_path)
            / meta.get("path", "").strip("/")
            / "00_overview.md"
        )
        if not overview.exists():
            return None
        try:
            return overview.read_text(encoding="utf-8")
        except Exception:
            return None

    return mcp
