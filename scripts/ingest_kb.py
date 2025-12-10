"""CLI for knowledge base ingestion."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import typer

from src.chunking import MarkdownChunker
from src.config import Settings
from src.embeddings import BGEM3Embedder
from src.ingestion import IngestionPipeline
from src.vectordb import LanceDBStore, SparseIndex

app = typer.Typer(help="HOI4 RAG ingestion commands.")


def _build_pipeline(settings: Settings) -> IngestionPipeline:
    chunker = MarkdownChunker()
    embedder = BGEM3Embedder(settings.embedding)
    lancedb_store = LanceDBStore(settings.paths.vectordb_path)
    sparse_index = SparseIndex(settings.paths.vectordb_path)
    return IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        lancedb_store=lancedb_store,
        sparse_index=sparse_index,
        config=settings,
    )


@app.command()  # type: ignore[untyped-decorator]
def ingest(
    source: Path = typer.Option(Path("./data/raw"), help="Path to knowledge base."),  # noqa: B008
    force: bool = typer.Option(False, help="Reindex everything."),  # noqa: B008
    validate: bool = typer.Option(False, help="Run validation after ingestion."),  # noqa: B008
) -> None:
    """Run full ingestion."""
    settings = Settings.from_env()
    pipeline = _build_pipeline(settings)
    typer.echo(f"Source: {source}")
    typer.echo(f"Vector DB: {settings.paths.vectordb_path}")

    start = perf_counter()
    stats = pipeline.ingest_full(source, force=force)
    total = perf_counter() - start

    typer.echo(
        f"Ingestion complete | docs: {stats.documents_processed} | "
        f"chunks: +{stats.chunks_created} | deleted: {stats.chunks_deleted} | "
        f"time: {total:.1f}s"
    )

    if validate:
        _validate(settings)


@app.command()  # type: ignore[untyped-decorator]
def validate() -> None:
    """Validate indexes exist."""
    settings = Settings.from_env()
    _validate(settings)


def _validate(settings: Settings) -> None:
    vectordb = Path(settings.paths.vectordb_path)
    table = vectordb / "hoi4_docs.lance"
    sparse = vectordb / "sparse_index.json"
    colbert = vectordb / "colbert_vectors.npz"

    issues = []
    if not table.exists():
        issues.append("LanceDB table missing (hoi4_docs.lance)")
    if not sparse.exists():
        issues.append("Sparse index missing (sparse_index.json)")
    if not colbert.exists():
        issues.append("ColBERT vectors missing (colbert_vectors.npz)")

    if issues:
        typer.echo("Validation issues:")
        for issue in issues:
            typer.echo(f"- {issue}")
        raise typer.Exit(code=1)

    typer.echo("Validation OK")


@app.command()  # type: ignore[untyped-decorator]
def stats() -> None:
    """Show simple ingestion stats (row counts)."""
    import lancedb

    settings = Settings.from_env()
    db = lancedb.connect(str(settings.paths.vectordb_path))
    if "hoi4_docs" not in db.table_names():
        typer.echo("No table hoi4_docs found.")
        raise typer.Exit(code=1)

    table = db.open_table("hoi4_docs")
    typer.echo(f"Total chunks: {len(table)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
