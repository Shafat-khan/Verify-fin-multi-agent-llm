"""
verifyfin.data_pipeline.chunker
==================================
Splits raw filing text into overlapping chunks suitable for embedding
and retrieval (Proposal Objective 2.3; Literature Review 2.3.4 --
semantic chunking that respects document structure outperforms naive
fixed-window chunking for RAG retrieval quality).

Three-level strategy:
    1. Split on SEC "Item N." section headers where present (e.g.
       "Item 1A. Risk Factors"), so a chunk never silently straddles
       two unrelated sections of a filing.
    2. Within each section, separate table rows (lines containing the
       " | " cell separator produced by EdgarClient's table-aware
       extraction) from prose. Each table row becomes its OWN chunk,
       verbatim -- NOT swept into a word-count sliding window.
    3. Prose blocks (and any table row, as a fallback, if step 2 finds
       none) go through a sliding word-count window with overlap.

Why table rows get their own chunk (step 2) -- a real retrieval-quality
bug found during development: a 10-Q's financial statement tables can
have dozens of line items. If many rows get packed into the same
512-word window, that chunk's embedding becomes an average over all of
them, and a specific row like "Total net sales | 254,940 | 230,901"
gets diluted alongside unrelated rows (inventory, tax, share counts...)
until its vector no longer points clearly at "revenue". This was
observed directly: claims about specific financial figures were
retrieving a generic, unrelated prose sentence as their best match,
because it repeated boilerplate context (the filing date) common to
every claim, while the actual relevant table row was diluted into a
low-signal chunk. Table rows are naturally short and self-contained,
so giving each one its own chunk removes the dilution entirely.

Note on "tokens": chunk_size_tokens / chunk_overlap_tokens in
config.yaml are treated here as an approximate word count, not exact
LLM tokens -- a reasonable simplification for MSc dissertation scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from verifyfin.utils.io_utils import load_config
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SEC_ITEM_HEADER_PATTERN = re.compile(r"(Item\s+\d+[A-Z]?\.\s)", re.IGNORECASE)


@dataclass
class Chunk:
    """A single retrievable unit of text, ready for embedding."""

    text: str
    chunk_index: int
    source_document_id: str
    source_type: str
    section_hint: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Chunker:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        vs_cfg = self.config["vector_store"]

        self.chunk_size_words = vs_cfg.get("chunk_size_tokens", 512)
        self.chunk_overlap_words = vs_cfg.get("chunk_overlap_tokens", 64)
        self.semantic_chunking = vs_cfg.get("semantic_chunking", True)

        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_words}) must be smaller than "
                f"chunk_size_tokens ({self.chunk_size_words})."
            )

    # ---------- section splitting ----------

    def _split_into_sections(self, text: str) -> list[tuple[Optional[str], str]]:
        """
        Splits filing text on 'Item N.' boundaries. Returns a list of
        (section_header_or_None, section_text) tuples. If semantic
        chunking is disabled, or no Item headers are found, returns the
        whole text as one section with no header.
        """
        if not self.semantic_chunking:
            return [(None, text)]

        parts = _SEC_ITEM_HEADER_PATTERN.split(text)
        if len(parts) <= 1:
            return [(None, text)]  # no Item headers found in this document

        sections: list[tuple[Optional[str], str]] = []
        preamble = parts[0].strip()
        if preamble:
            sections.append((None, preamble))

        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append((header, body.strip()))

        return sections

    # ---------- table row vs. prose splitting ----------

    @staticmethod
    def _split_table_and_prose_blocks(text: str) -> list[tuple[str, str]]:
        """
        Splits a block of text (typically one section) into an ordered
        list of ('table_row', line) and ('prose', block) tuples.

        A line is treated as a table row if it contains the " | " cell
        separator EdgarClient's table-aware extraction inserts between
        cells. Consecutive non-table lines are collapsed back into a
        single prose block, so ordinary paragraphs are unaffected.
        """
        lines = text.split("\n")
        blocks: list[tuple[str, str]] = []
        prose_buffer: list[str] = []

        for line in lines:
            if " | " in line:
                if prose_buffer:
                    blocks.append(("prose", " ".join(prose_buffer)))
                    prose_buffer = []
                stripped = line.strip()
                if stripped:
                    blocks.append(("table_row", stripped))
            else:
                stripped = line.strip()
                if stripped:
                    prose_buffer.append(stripped)

        if prose_buffer:
            blocks.append(("prose", " ".join(prose_buffer)))

        return blocks

    # ---------- sliding window ----------

    def _sliding_window(self, text: str) -> list[str]:
        """Splits a block of text into overlapping word-count windows."""
        words = text.split()
        if not words:
            return []
        if len(words) <= self.chunk_size_words:
            return [text.strip()]

        chunks = []
        step = self.chunk_size_words - self.chunk_overlap_words
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size_words]
            if not window:
                break
            chunks.append(" ".join(window))
            if start + self.chunk_size_words >= len(words):
                break
        return chunks

    # ---------- public API ----------

    def chunk_text(self, text: str, source_document_id: str, source_type: str) -> list[Chunk]:
        """
        Full chunking pipeline for one document: split into sections,
        then within each section separate table rows (each becoming its
        own chunk) from prose (which goes through the sliding window).
        """
        if not text or not text.strip():
            return []

        sections = self._split_into_sections(text)
        chunks: list[Chunk] = []
        idx = 0

        for section_header, section_text in sections:
            for block_type, block_text in self._split_table_and_prose_blocks(section_text):
                if block_type == "table_row":
                    chunks.append(
                        Chunk(
                            text=block_text,
                            chunk_index=idx,
                            source_document_id=source_document_id,
                            source_type=source_type,
                            section_hint=section_header,
                        )
                    )
                    idx += 1
                else:
                    for window_text in self._sliding_window(block_text):
                        chunks.append(
                            Chunk(
                                text=window_text,
                                chunk_index=idx,
                                source_document_id=source_document_id,
                                source_type=source_type,
                                section_hint=section_header,
                            )
                        )
                        idx += 1

        table_row_count = sum(1 for c in chunks if " | " in c.text)
        logger.info(
            f"Chunked document {source_document_id} into {len(chunks)} chunks "
            f"({table_row_count} table-row chunks, {len(chunks) - table_row_count} prose chunks, "
            f"{'semantic' if self.semantic_chunking else 'fixed-window'} mode)."
        )
        return chunks
