"""
Unit tests for verifyfin.data_pipeline.chunker. Pure logic, no external
services involved -- runs fully offline.
"""

import pytest
import yaml

from verifyfin.data_pipeline.chunker import Chunker
from verifyfin.utils.io_utils import load_config


@pytest.fixture
def chunker(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unused-for-this-test")
    return Chunker()


class TestSectionSplitting:
    def test_splits_on_item_headers(self, chunker):
        text = (
            "Preamble text. Item 1. Business This is the business section. "
            "Item 1A. Risk Factors These are the risks."
        )
        sections = chunker._split_into_sections(text)
        headers = [h for h, _ in sections if h is not None]
        assert any("Item 1." in h for h in headers)
        assert any("Item 1A." in h for h in headers)

    def test_no_headers_returns_single_section(self, chunker):
        text = "Just some plain text with no SEC item headers at all."
        sections = chunker._split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] is None

    def test_semantic_chunking_disabled_returns_whole_text_as_one_section(self, chunker):
        chunker.semantic_chunking = False
        text = "Item 1. Business text. Item 1A. Risk text."
        sections = chunker._split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] is None
        assert sections[0][1] == text


class TestSlidingWindow:
    def test_short_text_returns_single_chunk(self, chunker):
        text = "A short sentence."
        chunks = chunker._sliding_window(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_overlapping_chunks_with_correct_overlap(self, chunker):
        chunker.chunk_size_words = 10
        chunker.chunk_overlap_words = 3
        text = " ".join([f"word{i}" for i in range(25)])
        chunks = chunker._sliding_window(text)
        assert len(chunks) > 1

        chunk0_words = chunks[0].split()
        chunk1_words = chunks[1].split()
        # the last 3 words of chunk 0 must equal the first 3 words of chunk 1
        assert chunk0_words[-3:] == chunk1_words[:3]

    def test_empty_text_returns_no_chunks(self, chunker):
        assert chunker._sliding_window("") == []
        assert chunker._sliding_window("   ") == []


class TestChunkText:
    def test_empty_input_returns_empty_list(self, chunker):
        assert chunker.chunk_text("", "doc1", "edgar_10q") == []

    def test_chunks_carry_correct_metadata(self, chunker):
        text = "Some filing text about revenue and risk."
        chunks = chunker.chunk_text(text, source_document_id="doc123", source_type="edgar_10q")
        assert len(chunks) >= 1
        assert chunks[0].source_document_id == "doc123"
        assert chunks[0].source_type == "edgar_10q"
        assert chunks[0].chunk_index == 0

    def test_chunk_indices_are_sequential_across_sections(self, chunker):
        chunker.chunk_size_words = 5
        chunker.chunk_overlap_words = 1
        text = " ".join([f"word{i}" for i in range(30)])
        chunks = chunker.chunk_text(text, "doc1", "edgar_10q")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestTableRowIsolation:
    """
    Regression tests for a real retrieval-quality bug found during
    development: table rows swept into the same word-count sliding
    window as dozens of unrelated rows produced a diluted chunk
    embedding, causing claims about specific figures to retrieve an
    unrelated generic chunk instead of their actual source row.
    """

    def test_table_row_becomes_its_own_chunk(self, chunker):
        text = "Intro prose.\nTotal net sales | 254,940 | 230,901\nNet income | 71,675 | 68,201"
        chunks = chunker.chunk_text(text, "doc1", "edgar_10q")
        table_chunks = [c.text for c in chunks if " | " in c.text]
        assert "Total net sales | 254,940 | 230,901" in table_chunks
        assert "Net income | 71,675 | 68,201" in table_chunks

    def test_single_row_not_diluted_by_many_unrelated_rows(self, chunker):
        many_rows = "\n".join([f"Line item {i} | {i*1000} | {i*900}" for i in range(60)])
        text = "Total net sales | 254,940 | 230,901\n" + many_rows

        chunks = chunker.chunk_text(text, "doc1", "edgar_10q")
        revenue_chunk = next(c for c in chunks if "Total net sales" in c.text)

        assert revenue_chunk.text == "Total net sales | 254,940 | 230,901"
        assert "Line item" not in revenue_chunk.text  # must not be merged with unrelated rows

    def test_plain_text_with_no_table_rows_behaves_as_single_prose_block(self, chunker):
        # Text with no " | " anywhere must chunk exactly as before this fix.
        text = "Just some plain prose with no table structure at all."
        chunks = chunker.chunk_text(text, "doc1", "edgar_10q")
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_table_rows_and_prose_interleave_in_order(self, chunker):
        text = "Opening prose.\nRevenue | 100 | 90\nMore prose after the table."
        chunks = chunker.chunk_text(text, "doc1", "edgar_10q")
        texts = [c.text for c in chunks]
        assert texts == ["Opening prose.", "Revenue | 100 | 90", "More prose after the table."]


class TestValidation:
    def test_overlap_larger_than_chunk_size_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GROQ_API_KEY", "unused-for-this-test")
        cfg = load_config()
        cfg["vector_store"]["chunk_size_tokens"] = 10
        cfg["vector_store"]["chunk_overlap_tokens"] = 20  # invalid: overlap > size

        bad_config_path = tmp_path / "bad_config.yaml"
        with open(bad_config_path, "w") as f:
            yaml.safe_dump(cfg, f)

        with pytest.raises(ValueError, match="chunk_overlap_tokens"):
            Chunker(config_path=str(bad_config_path))
