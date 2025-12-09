from pathlib import Path

from hoi4_rag.chunking import MarkdownChunker


def test_chunk_document_extracts_frontmatter_and_sections(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text(
        "---\n"
        "domain: core\n"
        "concept: intro\n"
        "version: 1.0\n"
        "---\n"
        "# Title\n"
        "## Section One\n"
        "Paragraph text.\n"
        "```pdx\n"
        "code_block = yes\n"
        "```\n",
        encoding="utf-8",
    )

    chunker = MarkdownChunker()
    chunks = chunker.chunk_document(md)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.domain == "core"
    assert chunk.section_heading == "Section One"
    assert "code_block" in chunk.text
    assert chunk.section_anchor == "section-one"
