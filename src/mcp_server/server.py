"""FastMCP server exposing HOI4 RAG tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import Settings
from embeddings.bge_m3 import BGEM3Embedder
from retrieval import BGEReranker, HybridSearcher

from vectordb.lancedb_store import LanceDBStore
from vectordb.sparse_index import SparseIndex

try:  # Optional dependency guard for environments without MCP SDK
    from mcp.server import FastMCP

    if not hasattr(FastMCP, "tool") or not hasattr(FastMCP, "resource"):
        raise ImportError("mcp.server.FastMCP missing tool/resource decorators")
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
            raise ImportError(
                "mcp package not installed or incompatible. Install the 'mcp' package."
            )

    def tool(*args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator

    def resource(*args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator


def _load_master_index(kb_path: Path) -> dict[str, Any]:
    """Load master_index.json if present."""
    master_index = kb_path / "master_index.json"
    if not master_index.exists():
        raise ValueError("Master index not found")
    try:
        with master_index.open("r", encoding="utf-8") as f:
            return dict(json.load(f))
    except Exception as err:
        raise ValueError("Failed to load master index") from err


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
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
    if hasattr(result, "__dict__"):
        if isinstance(result, dict):  # dict subclass
            return {**result, **result.__dict__}
        return dict(result.__dict__)
    if isinstance(result, dict):  # plain dict
        return dict(result)
    return {k: getattr(result, k, None) for k in ("chunk_id", "text", "score", "metadata")}


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

    try:
        mcp = FastMCP(
            name=settings.mcp_server_name,
            description="HOI4 modding documentation retrieval system",
        )
    except TypeError:
        # Older/newer MCP versions may not accept description
        mcp = FastMCP(name=settings.mcp_server_name)

    @mcp.tool()
    def search_documentation(
        query: str, domain: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search HOI4 modding documentation for technical information."""
        filters = {"domain": domain} if domain else None
        initial = searcher.search(query, top_k=top_k * 2, filters=filters, use_colbert_rerank=False)
        reranked = reranker.rerank(query, initial, top_k=top_k)
        return [_result_to_dict(r) for r in reranked]

    @mcp.tool()
    def search_code_examples(
        query: str, language: str = "pdx", top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for working code examples and implementation patterns."""
        del language  # not yet used for filtering
        initial = searcher.search(query, top_k=top_k * 2, use_colbert_rerank=False)
        reranked = reranker.rerank(query, initial, top_k=top_k)
        return [_result_to_dict(r) for r in reranked]

    @mcp.tool()
    def get_document_section(
        document_id: str, include_context: bool = False
    ) -> dict[str, Any] | None:
        """Retrieve a specific documentation section by ID."""
        chunk = searcher.lancedb_store.get_chunk_by_id(document_id)
        if chunk is None:
            return None
        if not include_context:
            return _chunk_to_dict(chunk)

        # Include siblings from same file
        table = searcher.lancedb_store._require_table()
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
    def lookup_define(
        define_name: str, category: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Look up a specific HOI4 define value."""
        filters = {"domain": "defines_list"}
        query = define_name if category is None else f"{category} {define_name}"
        results = searcher.search(query, top_k=top_k, filters=filters, use_colbert_rerank=False)
        return [_result_to_dict(r) for r in results]

    @mcp.tool()
    def lookup_modifier(
        modifier_name: str, category: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Look up a specific HOI4 modifier."""
        filters = {"domain": "modifiers_list"}
        query = modifier_name if category is None else f"{category} {modifier_name}"
        results = searcher.search(query, top_k=top_k, filters=filters, use_colbert_rerank=False)
        return [_result_to_dict(r) for r in results]

    @mcp.tool()
    def diagnose_error(
        error_message: str, context: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Diagnose HOI4 modding errors from logs or descriptions."""
        combined = f"{error_message}\n{context}" if context else error_message
        results = searcher.search(combined, top_k=top_k, use_colbert_rerank=False)
        return [_result_to_dict(r) for r in results]

    @mcp.resource("hoi4://domains")
    def list_domains() -> list[dict[str, Any]]:
        """List available documentation domains."""
        index = _load_master_index(settings.paths.knowledge_base_path)
        domains = index.get("domains", {})
        return [{"name": name, "path": meta.get("path", "")} for name, meta in domains.items()]

    @mcp.resource("hoi4://domain/{domain_name}")
    def domain_overview(domain_name: str) -> str | None:
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
            return str(overview.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError("Failed to read overview") from err

    return mcp
