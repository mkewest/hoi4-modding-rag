"""Markdown-aware chunker for HOI4 RAG."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter


class ChunkingError(Exception):
    """Raised when markdown chunking fails."""


@dataclass
class Section:
    heading: str
    level: int
    content: str
    children: list[Section] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    text: str
    file_path: str
    domain: str
    concept: str
    version: str
    section_heading: str
    section_anchor: str
    parent_headings: list[str]
    token_count: int
    content_hash: str


class MarkdownChunker:
    """Markdown-aware chunking that preserves structure and code blocks."""

    def __init__(
        self,
        max_chunk_tokens: int = 1500,
        min_chunk_tokens: int = 100,
        overlap_tokens: int = 0,
        code_block_atomic: bool = True,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.code_block_atomic = code_block_atomic

    def chunk_document(self, file_path: Path) -> list[Chunk]:
        """Parse and chunk a single markdown file."""
        if not file_path.exists():
            raise ChunkingError(f"File not found: {file_path}")

        try:
            post = frontmatter.load(file_path)
        except Exception as exc:  # pragma: no cover - defensive
            raise ChunkingError(f"Failed to parse frontmatter for {file_path}") from exc

        metadata: dict[str, str] = {k: str(v) for k, v in post.metadata.items()}
        content: str = post.content
        domain = metadata.get("domain", "")
        concept = metadata.get("concept", "")
        version = metadata.get("version", "")

        sections = self._parse_sections(content)
        if not sections:
            return []

        chunks: list[Chunk] = []
        h2_sections = [s for s in sections if s.level == 2]
        target_sections = h2_sections if h2_sections else sections

        for section in target_sections:
            chunks.extend(
                self._section_to_chunks(
                    section=section,
                    file_path=file_path,
                    domain=domain,
                    concept=concept,
                    version=version,
                    parents=self._gather_parents(section, sections),
                )
            )

        return chunks

    def chunk_directory(self, dir_path: Path, recursive: bool = True) -> list[Chunk]:
        """Process all markdown files in a directory."""
        pattern = "**/*.md" if recursive else "*.md"
        all_chunks: list[Chunk] = []
        for path in sorted(dir_path.glob(pattern)):
            if path.name.lower() == "readme.md":
                continue
            all_chunks.extend(self.chunk_document(path))
        return all_chunks

    def _section_to_chunks(
        self,
        section: Section,
        file_path: Path,
        domain: str,
        concept: str,
        version: str,
        parents: list[str],
    ) -> list[Chunk]:
        text = self._render_section(section)
        token_count = self._estimate_tokens(text)

        if token_count <= self.max_chunk_tokens or not section.children:
            return [
                self._build_chunk(
                    text=text,
                    file_path=file_path,
                    domain=domain,
                    concept=concept,
                    version=version,
                    heading=section.heading,
                    parents=parents,
                )
            ]

        child_chunks: list[Chunk] = []
        for child in section.children:
            child_chunks.extend(
                self._section_to_chunks(
                    section=child,
                    file_path=file_path,
                    domain=domain,
                    concept=concept,
                    version=version,
                    parents=parents + [section.heading],
                )
            )

        if child_chunks:
            return child_chunks

        # Fallback: split by paragraphs if still too large
        paragraph_chunks: list[Chunk] = []
        for part in self._split_paragraphs(section.content):
            if not part.strip():
                continue
            paragraph_chunks.append(
                self._build_chunk(
                    text=part,
                    file_path=file_path,
                    domain=domain,
                    concept=concept,
                    version=version,
                    heading=section.heading,
                    parents=parents,
                )
            )
        return paragraph_chunks or [
            self._build_chunk(
                text=text,
                file_path=file_path,
                domain=domain,
                concept=concept,
                version=version,
                heading=section.heading,
                parents=parents,
            )
        ]

    def _build_chunk(
        self,
        text: str,
        file_path: Path,
        domain: str,
        concept: str,
        version: str,
        heading: str,
        parents: list[str],
    ) -> Chunk:
        anchor = self._generate_anchor(heading)
        chunk_id = f"{file_path.as_posix()}#{anchor}"
        token_count = self._estimate_tokens(text)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return Chunk(
            id=chunk_id,
            text=text,
            file_path=file_path.as_posix(),
            domain=domain,
            concept=concept,
            version=version,
            section_heading=heading,
            section_anchor=anchor,
            parent_headings=parents,
            token_count=token_count,
            content_hash=content_hash,
        )

    def _parse_sections(self, content: str) -> list[Section]:
        """Parse markdown into a hierarchy of sections."""
        lines = content.splitlines()
        root_sections: list[Section] = []
        stack: list[Section] = []
        in_code_block = False
        current_code: list[str] = []

        def start_section(level: int, heading: str) -> Section:
            new_sec = Section(heading=heading.strip(), level=level, content="")
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                stack[-1].children.append(new_sec)
            else:
                root_sections.append(new_sec)
            stack.append(new_sec)
            return new_sec

        current_section: Section = start_section(1, "Introduction")

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_code.append(line)
                current_section.content += line + "\n"
                if not in_code_block:
                    current_section.code_blocks.append("\n".join(current_code))
                    current_code = []
                continue

            if in_code_block:
                current_code.append(line)
                current_section.content += line + "\n"
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                current_section = start_section(level, heading_text)
                continue

            current_section.content += line + "\n"

        return root_sections

    def _render_section(self, section: Section) -> str:
        """Render section text including children."""
        parts = [f"{'#' * section.level} {section.heading}".strip(), section.content.strip()]
        for child in section.children:
            parts.append(self._render_section(child))
        return "\n\n".join(part for part in parts if part)

    def _split_paragraphs(self, text: str) -> Iterable[str]:
        """Split text into paragraphs to fit size constraints."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            if self._estimate_tokens(para) <= self.max_chunk_tokens:
                yield para
                continue
            # Further split oversized paragraphs by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf: list[str] = []
            for sent in sentences:
                buf.append(sent)
                candidate = " ".join(buf)
                if self._estimate_tokens(candidate) >= self.max_chunk_tokens:
                    yield candidate
                    buf = []
            if buf:
                yield " ".join(buf)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using a simple heuristic."""
        return max(1, len(text) // 4)

    @staticmethod
    def _generate_anchor(heading: str) -> str:
        """Generate URL-safe anchor from heading."""
        anchor = heading.lower().strip()
        anchor = re.sub(r"[^\w\s-]", "", anchor)
        anchor = re.sub(r"\s+", "-", anchor)
        anchor = re.sub(r"-+", "-", anchor)
        return anchor.strip("-")

    @staticmethod
    def _gather_parents(target: Section, roots: list[Section]) -> list[str]:
        """Return parent headings for a given section."""
        path: list[str] = []

        def dfs(node: Section, parents: list[str]) -> bool:
            if node is target:
                path.extend(parents)
                return True
            for child in node.children:
                if dfs(child, parents + [node.heading]):
                    return True
            return False

        for root in roots:
            if dfs(root, []):
                break
        return path
