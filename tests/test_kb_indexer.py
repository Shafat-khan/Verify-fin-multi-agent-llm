"""
Unit tests for verifyfin.data_pipeline.kb_indexer.

FredClient, NewsClient, and VectorStore are all mocked -- these tests
run offline.
"""

from unittest.mock import MagicMock

import pytest

from verifyfin.data_pipeline.kb_indexer import index_macro_data, index_news_headlines
from verifyfin.data_pipeline.news_client import NewsUnavailableError


class TestIndexMacroData:
    def test_indexes_full_historical_series_not_just_latest_value(self):
        mock_store = MagicMock()
        mock_store.add_chunks.return_value = 6

        mock_fred = MagicMock()
        mock_fred.default_series = ["FEDFUNDS", "UNRATE"]
        mock_fred.get_series.side_effect = lambda series_id: {
            "FEDFUNDS": {"dates": ["2026-05-01", "2026-06-01", "2026-07-01"], "values": [5.25, 5.20, 3.63]},
            "UNRATE": {"dates": ["2026-06-01", "2026-07-01"], "values": [4.1, 4.2]},
        }[series_id]

        added = index_macro_data(mock_store, "AAPL", fred_client=mock_fred)

        assert added == 6
        call_kwargs = mock_store.add_chunks.call_args
        chunks = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["chunks"]
        assert len(chunks) == 5  # 3 FEDFUNDS observations + 2 UNRATE observations, not just 1 "latest" each
        assert any("5.25" in c.text for c in chunks)  # an OLDER observation, not just the latest

    def test_uses_ticker_for_indexing(self):
        mock_store = MagicMock()
        mock_fred = MagicMock()
        mock_fred.default_series = ["FEDFUNDS"]
        mock_fred.get_series.return_value = {"dates": ["2026-07-01"], "values": [3.63]}

        index_macro_data(mock_store, "AAPL", fred_client=mock_fred)

        call_args = mock_store.add_chunks.call_args
        assert call_args.kwargs.get("ticker") == "AAPL" or "AAPL" in call_args.args

    def test_series_fetch_failure_is_skipped_not_fatal(self):
        mock_store = MagicMock()
        mock_store.add_chunks.return_value = 1
        mock_fred = MagicMock()
        mock_fred.default_series = ["FEDFUNDS", "BROKEN_SERIES"]

        def side_effect(series_id):
            if series_id == "BROKEN_SERIES":
                raise Exception("series not found")
            return {"dates": ["2026-07-01"], "values": [3.63]}

        mock_fred.get_series.side_effect = side_effect

        added = index_macro_data(mock_store, "AAPL", fred_client=mock_fred)  # must not raise
        assert added == 1


class TestIndexNewsHeadlines:
    def test_indexes_headline_titles_independently_of_any_claim(self):
        mock_store = MagicMock()
        mock_store.add_chunks.return_value = 2

        mock_news = MagicMock()
        mock_news.get_recent_headlines.return_value = [
            {"title": "Apple beats earnings", "link": "https://example.com/1", "published": "", "source": "Reuters"},
            {"title": "Apple stock rises", "link": "https://example.com/2", "published": "", "source": "CNBC"},
        ]

        added = index_news_headlines(mock_store, "AAPL", news_client=mock_news)

        assert added == 2
        call_args = mock_store.add_chunks.call_args
        chunks = call_args.args[0] if call_args.args else call_args.kwargs["chunks"]
        assert chunks[0].text == "Apple beats earnings"  # the raw headline, not a claim's paraphrase
        assert chunks[1].text == "Apple stock rises"

    def test_no_news_available_returns_zero_not_error(self):
        mock_store = MagicMock()
        mock_news = MagicMock()
        mock_news.get_recent_headlines.side_effect = NewsUnavailableError("no news")

        added = index_news_headlines(mock_store, "AAPL", news_client=mock_news)  # must not raise
        assert added == 0
        mock_store.add_chunks.assert_not_called()


class TestIndexMacroDataAsOf:
    def _fred(self):
        mock_fred = MagicMock()
        mock_fred.default_series = ["FEDFUNDS"]
        mock_fred.get_series.return_value = {"dates": ["2026-02-01", "2026-03-01", "2026-04-01", "2026-09-01"],
                                             "values": [3.6, 3.64, 3.64, 3.75]}
        return mock_fred

    def test_observations_after_as_of_are_not_indexed(self):
        mock_store = MagicMock()
        index_macro_data(mock_store, "AAPL", fred_client=self._fred(), as_of="2026-03-28")
        chunks = mock_store.add_chunks.call_args.args[0]
        assert [c.text for c in chunks] == ["FEDFUNDS observation: 3.6 on 2026-02-01", "FEDFUNDS observation: 3.64 on 2026-03-01"]

    def test_as_of_none_indexes_everything_as_before(self):
        mock_store = MagicMock()
        index_macro_data(mock_store, "AAPL", fred_client=self._fred())
        assert len(mock_store.add_chunks.call_args.args[0]) == 4

    def test_an_observation_on_the_as_of_date_is_kept(self):
        mock_store = MagicMock()
        index_macro_data(mock_store, "AAPL", fred_client=self._fred(), as_of="2026-04-01")
        assert len(mock_store.add_chunks.call_args.args[0]) == 3
