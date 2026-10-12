"""
Unit tests for verifyfin.agents.fundamental_agent.

EdgarXbrlClient and GroqClient are mocked at construction -- these
tests run offline with zero API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from verifyfin.agents.fundamental_agent import FundamentalAgent
from verifyfin.schemas.claim import ClaimCategory, GroundingVerdict

FAKE_FUNDAMENTALS = {
    "ticker": "AAPL",
    "revenue": {
        "value": 95000000000, "unit": "USD", "period_end": "2026-03-28",
        "form": "10-Q", "xbrl_tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
        "accession_number": "0000320193-26-000013",
    },
    "net_income": {
        "value": 71675000000, "unit": "USD", "period_end": "2026-03-28",
        "form": "10-Q", "xbrl_tag": "NetIncomeLoss",
        "accession_number": "0000320193-26-000013",
    },
    "eps_basic": None,
    "eps_diluted": None,
    "total_assets": None,
    "total_liabilities": None,
    "stockholders_equity": None,
    "operating_income": None,
}

VALID_LLM_RESPONSE = json.dumps(
    [
        {"concept": "revenue", "claim_text": "AAPL reported total revenue of 95000000000 USD as of 2026-03-28."},
        {"concept": "net_income", "claim_text": "AAPL reported net income of 71675000000 USD as of 2026-03-28."},
    ]
)


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unused-for-this-test")
    with patch("verifyfin.agents.fundamental_agent.EdgarXbrlClient") as mock_xbrl_cls, \
         patch("verifyfin.agents.fundamental_agent.GroqClient") as mock_groq_cls:
        mock_xbrl = MagicMock()
        mock_groq = MagicMock()
        mock_xbrl_cls.return_value = mock_xbrl
        mock_groq_cls.return_value = mock_groq

        a = FundamentalAgent()

    return a, mock_xbrl, mock_groq


class TestAnalyzeHappyPath:
    def test_produces_one_claim_per_available_concept(self, agent):
        a, mock_xbrl, mock_groq = agent
        mock_xbrl.get_latest_fundamentals.return_value = FAKE_FUNDAMENTALS
        mock_groq.complete.return_value = VALID_LLM_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")

        assert len(claims) == 2  # revenue, net_income -- the only 2 concepts with data
        assert all(c.claim_category == ClaimCategory.NUMERICAL for c in claims)
        assert all(c.verdict == GroundingVerdict.UNVERIFIED for c in claims)  # not yet verified

    def test_source_reference_uses_real_accession_number(self, agent):
        a, mock_xbrl, mock_groq = agent
        mock_xbrl.get_latest_fundamentals.return_value = FAKE_FUNDAMENTALS
        mock_groq.complete.return_value = VALID_LLM_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")
        revenue_claim = next(c for c in claims if "revenue" in c.claim_text.lower())
        assert revenue_claim.source_reference.document_id == "0000320193-26-000013"
        assert revenue_claim.source_reference.source_type == "edgar_xbrl"

    def test_confidence_is_high_for_xbrl_sourced_claims(self, agent):
        a, mock_xbrl, mock_groq = agent
        mock_xbrl.get_latest_fundamentals.return_value = FAKE_FUNDAMENTALS
        mock_groq.complete.return_value = VALID_LLM_RESPONSE

        claims = a.analyze("AAPL", "2026-Q1")
        assert all(c.confidence_score == 0.95 for c in claims)


class TestLLMResponseRobustness:
    def test_markdown_fenced_json_is_parsed_correctly(self, agent):
        a, mock_xbrl, mock_groq = agent
        mock_xbrl.get_latest_fundamentals.return_value = FAKE_FUNDAMENTALS
        mock_groq.complete.return_value = f"```json\n{VALID_LLM_RESPONSE}\n```"

        claims = a.analyze("AAPL", "2026-Q1")
        assert len(claims) == 2  # both concepts phrased correctly despite the fence

    def test_malformed_json_falls_back_to_deterministic_phrasing(self, agent):
        a, mock_xbrl, mock_groq = agent
        mock_xbrl.get_latest_fundamentals.return_value = FAKE_FUNDAMENTALS
        mock_groq.complete.return_value = "this is not valid json at all {{{"

        claims = a.analyze("AAPL", "2026-Q1")
        assert len(claims) == 2  # claims still produced, via deterministic fallback
        assert all("as of" in c.claim_text for c in claims)  # matches the fallback template

    def test_partial_llm_response_falls_back_for_missing_concept_only(self, agent):
        a, mock_xbrl, mock_groq = agent
        mock_xbrl.get_latest_fundamentals.return_value = FAKE_FUNDAMENTALS
        partial_response = json.dumps([{"concept": "revenue", "claim_text": "Revenue was high."}])
        mock_groq.complete.return_value = partial_response

        claims = a.analyze("AAPL", "2026-Q1")
        revenue_claim = next(c for c in claims if c.source_reference.excerpt.startswith("RevenueFromContract"))
        net_income_claim = next(c for c in claims if c.source_reference.excerpt.startswith("NetIncomeLoss"))
        assert revenue_claim.claim_text == "Revenue was high."  # from the LLM
        assert "as of" in net_income_claim.claim_text  # deterministic fallback


class TestNoDataAvailable:
    def test_no_xbrl_concepts_returns_no_claims_and_skips_llm_call(self, agent):
        a, mock_xbrl, mock_groq = agent
        mock_xbrl.get_latest_fundamentals.return_value = {"ticker": "AAPL"}  # nothing else

        claims = a.analyze("AAPL", "2026-Q1")
        assert claims == []
        mock_groq.complete.assert_not_called()  # must not waste an LLM call with nothing to phrase
