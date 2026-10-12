"""
Unit tests for verifyfin.data_pipeline.vector_store.

Both chromadb and sentence-transformers are mocked -- these tests run
offline without downloading the real embedding model or creating a
real Chroma database.
"""

from unittest.mock import MagicMock, patch

import pytest

from verifyfin.data_pipeline.chunker import Chunk
from verifyfin.data_pipeline.vector_store import VectorStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unused-for-this-test")

    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_embedder = MagicMock()
    mock_embedder.encode.side_effect = lambda docs, show_progress_bar=False: MagicMock(
        tolist=lambda: [[0.1, 0.2, 0.3] for _ in docs]
    )

    with patch("verifyfin.data_pipeline.vector_store.chromadb.PersistentClient", return_value=mock_client), \
         patch("verifyfin.data_pipeline.vector_store.SentenceTransformer", return_value=mock_embedder):
        vs = VectorStore()

    return vs, mock_collection, mock_embedder


class TestAddChunks:
    def test_empty_chunks_list_returns_zero_and_skips_upsert(self, store):
        vs, mock_collection, _ = store
        assert vs.add_chunks([], "AAPL") == 0
        mock_collection.upsert.assert_not_called()

    def test_adds_chunks_and_returns_count(self, store):
        vs, mock_collection, _ = store
        chunks = [
            Chunk(text="Revenue grew.", chunk_index=0, source_document_id="doc1", source_type="edgar_10q"),
            Chunk(text="Risk factors include...", chunk_index=1, source_document_id="doc1", source_type="edgar_10q"),
        ]
        count = vs.add_chunks(chunks, "AAPL")
        assert count == 2
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert len(call_kwargs["ids"]) == 2
        assert call_kwargs["ids"][0] == "AAPL_doc1_0"

    def test_metadata_includes_ticker_and_section_hint(self, store):
        vs, mock_collection, _ = store
        chunks = [
            Chunk(
                text="Text", chunk_index=0, source_document_id="doc1",
                source_type="edgar_10q", section_hint="Item 1A.",
            )
        ]
        vs.add_chunks(chunks, "AAPL")
        metadatas = mock_collection.upsert.call_args.kwargs["metadatas"]
        assert metadatas[0]["ticker"] == "AAPL"
        assert metadatas[0]["section_hint"] == "Item 1A."


class TestQuery:
    def test_converts_cosine_distance_to_similarity(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.return_value = {
            "documents": [["Revenue grew 12%."]],
            "metadatas": [[{"ticker": "AAPL"}]],
            "distances": [[0.15]],
        }
        results = vs.query("What was the revenue growth?")
        assert len(results) == 1
        assert results[0]["similarity_score"] == 0.85

    def test_applies_ticker_filter_and_top_k(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        vs.query("test query", ticker="AAPL", top_k=3)
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"ticker": "AAPL"}
        assert call_kwargs["n_results"] == 3

    def test_no_ticker_filter_passes_none(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        vs.query("test query")
        assert mock_collection.query.call_args.kwargs["where"] is None

    def test_where_document_contains_passes_chroma_contains_filter(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        vs.query("test query", where_document_contains="254,940")
        assert mock_collection.query.call_args.kwargs["where_document"] == {"$contains": "254,940"}

    def test_no_where_document_contains_passes_none(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        vs.query("test query")
        assert mock_collection.query.call_args.kwargs["where_document"] is None


class TestCount:
    def test_returns_collection_count(self, store):
        vs, mock_collection, _ = store
        mock_collection.count.return_value = 42
        assert vs.count() == 42


class TestRetrievalInstrumentation:
    def test_query_count_increments_per_call(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        vs.query("query 1")
        vs.query("query 2")

        assert vs.get_stats()["query_count"] == 2

    def test_query_latency_is_recorded(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        vs.query("query 1")

        assert vs.get_stats()["total_query_time_seconds"] >= 0.0

    def test_latency_still_recorded_even_if_query_raises(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.side_effect = Exception("chroma error")

        with pytest.raises(Exception):
            vs.query("query 1")

        # query_count/timing must still be recorded even on failure (finally block),
        # so a failing scenario isn't silently invisible in the metrics.
        assert vs.get_stats()["query_count"] == 1

    def test_reset_stats_zeroes_counters(self, store):
        vs, mock_collection, _ = store
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        vs.query("query 1")
        vs.reset_stats()

        stats = vs.get_stats()
        assert stats["query_count"] == 0
        assert stats["total_query_time_seconds"] == 0.0
