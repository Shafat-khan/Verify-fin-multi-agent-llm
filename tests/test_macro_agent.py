"""
Unit tests for verifyfin.agents.macro_agent.

FredClient and GroqClient are mocked -- these tests run offline with
zero API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from verifyfin.agents.macro_agent import MacroAgent
from verifyfin.schemas.claim import AgentRole, ClaimCategory

FAKE_SNAPSHOT = {
    "FEDFUNDS": {"series_id": "FEDFUNDS", "latest_value": 3.63, "latest_date": "2026-06-01"},
    "UNRATE": {"series_id": "UNRATE", "latest_value": 4.2, "latest_date": "2026-06-01"},
}

VALID_LLM_RESPONSE = json.dumps(
    [
        {"series_id": "FEDFUNDS", "claim_text": "The Federal Funds Rate was 3.63% as of June 1, 2026."},
        {"series_id": "UNRATE", "claim_text": "The U.S. unemployment rate was 4.2% as of June 1, 2026."},
    ]
)


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "unused-for-this-test")
    with patch("verifyfin.agents.macro_agent.FredClient") as mock_fred_cls, \
         patch("verifyfin.agents.macro_agent.GroqClient") as mock_groq_cls:
        mock_fred = MagicMock()
        mock_groq = MagicMock()
        mock_fred_cls.return_value = mock_fred
        mock_groq_cls.return_value = mock_groq

        a = MacroAgent()

    return a, mock_fred, mock_groq


class TestAnalyzeHappyPath:
    def test_produces_one_claim_per_series(self, agent):
        a, mock_fred, mock_groq = agent
        mock_fred.get_macro_snapshot.return_value = FAKE_SNAPSHOT
        mock_groq.complete.return_value = VALID_LLM_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")

        assert len(claims) == 2
        assert all(c.claim_category == ClaimCategory.MACRO for c in claims)
        assert all(c.produced_by == AgentRole.MACRO for c in claims)

    def test_source_reference_uses_series_id(self, agent):
        a, mock_fred, mock_groq = agent
        mock_fred.get_macro_snapshot.return_value = FAKE_SNAPSHOT
        mock_groq.complete.return_value = VALID_LLM_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")
        fed_claim = next(c for c in claims if c.source_reference.document_id == "FEDFUNDS")
        assert fed_claim.source_reference.source_type == "fred"
        assert "3.63" in fed_claim.source_reference.excerpt

    def test_confidence_is_high_for_fred_sourced_claims(self, agent):
        a, mock_fred, mock_groq = agent
        mock_fred.get_macro_snapshot.return_value = FAKE_SNAPSHOT
        mock_groq.complete.return_value = VALID_LLM_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")
        assert all(c.confidence_score == 0.95 for c in claims)


class TestLLMResponseRobustness:
    def test_markdown_fenced_json_is_parsed_correctly(self, agent):
        a, mock_fred, mock_groq = agent
        mock_fred.get_macro_snapshot.return_value = FAKE_SNAPSHOT
        mock_groq.complete.return_value = f"```json\n{VALID_LLM_RESPONSE}\n```"

        claims = a.analyze("AAPL", "2026-Q1")
        assert len(claims) == 2

    def test_malformed_json_falls_back_to_deterministic_phrasing(self, agent):
        a, mock_fred, mock_groq = agent
        mock_fred.get_macro_snapshot.return_value = FAKE_SNAPSHOT
        mock_groq.complete.return_value = "not valid json {{{"

        claims = a.analyze("AAPL", "2026-Q1")
        assert len(claims) == 2
        assert all("per FRED" in c.claim_text for c in claims)  # matches the fallback template


class TestNoDataAvailable:
    def test_empty_snapshot_returns_no_claims_and_skips_llm_call(self, agent):
        a, mock_fred, mock_groq = agent
        mock_fred.get_macro_snapshot.return_value = {}

        claims = a.analyze("AAPL", "2026-Q1")
        assert claims == []
        mock_groq.complete.assert_not_called()


class TestNoLookAhead:
    """An analysis 'as of' a period end must not use an observation dated after it (v3 fix)."""

    SERIES = {"dates": ["2026-01-01", "2026-03-01", "2026-04-01", "2026-05-01"], "values": [3.64, 3.64, 3.7, 3.8]}

    @staticmethod
    def _setup(agent, series):
        a, mock_fred, mock_groq = agent
        mock_fred.default_series = ["FEDFUNDS"]
        mock_fred.get_series.return_value = series
        mock_groq.complete.return_value = json.dumps([{"series_id": "FEDFUNDS", "claim_text": "x"}])
        return a, mock_groq

    def test_quarter_ending_28_march_uses_the_march_observation_not_the_april_one(self, agent):
        a, _ = self._setup(agent, self.SERIES)
        data = a._get_series_data("2026-03-28")
        assert data["FEDFUNDS"]["latest_date"] == "2026-03-01"          # the old 'closest' rule picked 2026-04-01 (4 days later)

    def test_observation_on_the_target_date_itself_is_allowed(self, agent):
        a, _ = self._setup(agent, self.SERIES)
        assert a._get_series_data("2026-04-01")["FEDFUNDS"]["latest_date"] == "2026-04-01"

    def test_never_returns_a_date_after_the_target(self, agent):
        a, _ = self._setup(agent, self.SERIES)
        for target in ("2026-01-01", "2026-02-15", "2026-03-31", "2026-04-30", "2027-01-01"):
            got = a._get_series_data(target)["FEDFUNDS"]["latest_date"]
            assert got <= target

    def test_target_before_the_first_observation_gives_no_claim_for_that_series(self, agent):
        a, mock_groq = self._setup(agent, self.SERIES)
        assert a._get_series_data("2025-06-01") == {}
        assert a.analyze("AAPL", "2025-Q2", target_date="2025-06-01") == []
        mock_groq.complete.assert_not_called()

    def test_helper_handles_unsorted_dates(self):
        idx = MacroAgent._latest_observation_index_on_or_before(["2026-04-01", "2026-01-01", "2026-03-01"], "2026-03-28")
        assert idx == 2
