"""
Unit tests for verifyfin.data_pipeline.news_client.

feedparser is mocked -- these tests run offline.
"""

from unittest.mock import MagicMock, patch

import pytest

from verifyfin.data_pipeline.news_client import NewsClient, NewsUnavailableError


def _fake_feed(num_entries: int = 3, bozo: int = 0):
    feed = MagicMock()
    feed.bozo = bozo
    feed.entries = [
        {
            "title": f"AAPL headline {i}",
            "link": f"https://news.example.com/article-{i}",
            "published": "Mon, 01 Aug 2026 12:00:00 GMT",
            "source": {"title": "Example News"},
        }
        for i in range(num_entries)
    ]
    return feed


@pytest.fixture
def client(tmp_path):
    c = NewsClient()
    c.cache_dir = tmp_path / "news_cache"
    c.cache_dir.mkdir()
    return c


class TestGetRecentHeadlines:
    def test_returns_expected_number_of_articles(self, client):
        with patch("verifyfin.data_pipeline.news_client.feedparser.parse", return_value=_fake_feed(5)):
            articles = client.get_recent_headlines("AAPL", max_articles=5)
        assert len(articles) == 5
        assert articles[0]["title"] == "AAPL headline 0"
        assert articles[0]["source"] == "Example News"

    def test_respects_max_articles_limit(self, client):
        with patch("verifyfin.data_pipeline.news_client.feedparser.parse", return_value=_fake_feed(10)):
            articles = client.get_recent_headlines("AAPL", max_articles=3)
        assert len(articles) == 3

    def test_second_call_uses_cache(self, client):
        with patch("verifyfin.data_pipeline.news_client.feedparser.parse", return_value=_fake_feed(3)) as mock_parse:
            client.get_recent_headlines("AAPL")
            client.get_recent_headlines("AAPL")
        assert mock_parse.call_count == 1

    def test_no_entries_and_bozo_raises(self, client):
        empty_feed = _fake_feed(0, bozo=1)
        empty_feed.entries = []
        with patch("verifyfin.data_pipeline.news_client.feedparser.parse", return_value=empty_feed):
            with pytest.raises(NewsUnavailableError):
                client.get_recent_headlines("NOTATICKER")

    def test_no_entries_without_bozo_still_raises(self, client):
        """Even a 'successfully parsed' feed with zero entries is unusable -- must raise, not return []."""
        empty_feed = _fake_feed(0, bozo=0)
        empty_feed.entries = []
        with patch("verifyfin.data_pipeline.news_client.feedparser.parse", return_value=empty_feed):
            with pytest.raises(NewsUnavailableError):
                client.get_recent_headlines("NOTATICKER")

    def test_uses_company_name_in_query_when_provided(self, client):
        with patch("verifyfin.data_pipeline.news_client.feedparser.parse", return_value=_fake_feed(1)) as mock_parse:
            client.get_recent_headlines("AAPL", company_name="Apple Inc.")
        called_url = mock_parse.call_args.args[0]
        assert "Apple+Inc" in called_url or "Apple%20Inc" in called_url
