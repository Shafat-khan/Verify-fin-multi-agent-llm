"""
Unit tests for verifyfin.agents.sentiment_agent.

NewsClient and GroqClient are mocked -- these tests run offline with
zero API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from verifyfin.agents.sentiment_agent import SentimentAgent
from verifyfin.data_pipeline.news_client import NewsUnavailableError
from verifyfin.schemas.claim import AgentRole, ClaimCategory

FAKE_ARTICLES = [
    {"title": "Apple beats earnings expectations", "link": "https://news.example.com/1", "published": "Mon, 01 Aug 2026", "source": "Reuters"},
    {"title": "Apple announces new product line", "link": "https://news.example.com/2", "published": "Tue, 02 Aug 2026", "source": "Bloomberg"},
    {"title": "Analysts raise Apple price target", "link": "https://news.example.com/3", "published": "Wed, 03 Aug 2026", "source": "CNBC"},
]

VALID_RESPONSE = json.dumps(
    {
        "sentiment": "positive",
        "claim_text": "Recent news coverage of AAPL is positive, driven by strong earnings and raised price targets.",
        "supporting_headline_indices": [0, 2],
        "confidence": 0.85,
    }
)


@pytest.fixture
def agent():
    with patch("verifyfin.agents.sentiment_agent.NewsClient") as mock_news_cls:
        mock_news = MagicMock()
        mock_news_cls.return_value = mock_news
        mock_llm = MagicMock()
        a = SentimentAgent(llm_client=mock_llm)
    return a, mock_news, mock_llm


class TestAnalyzeHappyPath:
    def test_produces_one_claim(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = VALID_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")

        assert len(claims) == 1
        assert claims[0].claim_category == ClaimCategory.SENTIMENT
        assert claims[0].produced_by == AgentRole.SENTIMENT

    def test_confidence_reflects_llm_self_report_not_fixed_value(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = VALID_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")
        assert claims[0].confidence_score == 0.85  # not 0.95 like XBRL-sourced claims

    def test_source_reference_cites_supporting_headlines(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = VALID_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")
        excerpt = claims[0].source_reference.excerpt
        assert "Apple beats earnings expectations" in excerpt
        assert "Analysts raise Apple price target" in excerpt
        assert "Apple announces new product line" not in excerpt  # index 1 was NOT in supporting indices

    def test_document_id_is_a_real_article_link(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = VALID_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")
        assert claims[0].source_reference.document_id == "https://news.example.com/1"


class TestResponseRobustness:
    def test_markdown_fenced_json_is_parsed_correctly(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = f"```json\n{VALID_RESPONSE}\n```"

        claims = a.analyze("AAPL", "2026-Q1")
        assert len(claims) == 1

    def test_malformed_json_produces_no_claim_not_a_guess(self, agent):
        """Unlike numeric agents, there's no safe fallback for an interpretive judgment -- must skip, not guess."""
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = "not valid json {{{"

        claims = a.analyze("AAPL", "2026-Q1")
        assert claims == []

    def test_unrecognised_sentiment_value_produces_no_claim(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = json.dumps(
            {"sentiment": "very good", "claim_text": "x", "supporting_headline_indices": [0], "confidence": 0.8}
        )

        claims = a.analyze("AAPL", "2026-Q1")
        assert claims == []

    def test_invalid_confidence_falls_back_to_default_not_crash(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = json.dumps(
            {"sentiment": "positive", "claim_text": "x", "supporting_headline_indices": [0], "confidence": "high"}
        )

        claims = a.analyze("AAPL", "2026-Q1")
        assert len(claims) == 1
        assert 0.0 <= claims[0].confidence_score <= 1.0

    def test_out_of_range_indices_are_filtered_not_crash(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.return_value = FAKE_ARTICLES
        mock_llm.complete.return_value = json.dumps(
            {"sentiment": "positive", "claim_text": "x", "supporting_headline_indices": [0, 99], "confidence": 0.7}
        )

        claims = a.analyze("AAPL", "2026-Q1")  # must not raise IndexError
        assert len(claims) == 1


class TestNoNewsAvailable:
    def test_no_news_returns_empty_list_not_error(self, agent):
        a, mock_news, mock_llm = agent
        mock_news.get_recent_headlines.side_effect = NewsUnavailableError("no news found")

        claims = a.analyze("AAPL", "2026-Q1")

        assert claims == []
        mock_llm.complete.assert_not_called()
