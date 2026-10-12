"""
Unit tests for verifyfin.agents.debate_agent.

GroqClient is mocked -- these tests run offline with zero API calls.
"""

import json
from unittest.mock import MagicMock

import pytest

from verifyfin.agents.debate_agent import DebateAgent
from verifyfin.schemas.claim import (
    AgentRole,
    ClaimCategory,
    ClaimObject,
    GroundingVerdict,
    SourceReference,
)


def make_claim(verdict: GroundingVerdict, retrieved_evidence: str = "Some evidence text.") -> ClaimObject:
    claim = ClaimObject(
        claim_text="Apple reported total revenue of $254,940,000,000.",
        claim_category=ClaimCategory.NUMERICAL,
        produced_by=AgentRole.FUNDAMENTAL,
        source_reference=SourceReference(source_type="edgar_xbrl", document_id="0000320193-26-000013"),
        confidence_score=0.95,
        ticker="AAPL",
        time_period="2026-Q1",
    )
    claim.verdict = verdict
    claim.retrieved_evidence = retrieved_evidence
    return claim


@pytest.fixture
def agent():
    mock_llm = MagicMock()
    a = DebateAgent(llm_client=mock_llm)
    return a, mock_llm


class TestOnlyDebatesChallenged:
    def test_grounded_claim_is_untouched_and_no_llm_call_made(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.GROUNDED)

        result = a.debate_claim(claim)

        assert result.debate_resolution is None
        mock_llm.complete.assert_not_called()

    def test_blocked_claim_is_untouched_and_no_llm_call_made(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.BLOCKED)

        result = a.debate_claim(claim)

        assert result.debate_resolution is None
        mock_llm.complete.assert_not_called()

    def test_challenged_claim_triggers_llm_call(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.CHALLENGED)
        mock_llm.complete.return_value = json.dumps({"verdict": "upheld", "reasoning": "Matches exactly."})

        a.debate_claim(claim)

        mock_llm.complete.assert_called_once()


class TestVerdictParsing:
    def test_upheld_verdict_makes_claim_usable(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.CHALLENGED)
        mock_llm.complete.return_value = json.dumps({"verdict": "upheld", "reasoning": "Numbers match."})

        result = a.debate_claim(claim)

        assert result.debate_resolution == "upheld"
        assert result.is_usable_in_final_report() is True

    def test_rejected_verdict_keeps_claim_unusable(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.CHALLENGED)
        mock_llm.complete.return_value = json.dumps({"verdict": "rejected", "reasoning": "Different period."})

        result = a.debate_claim(claim)

        assert result.debate_resolution == "rejected"
        assert result.is_usable_in_final_report() is False

    def test_markdown_fenced_json_is_parsed_correctly(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.CHALLENGED)
        mock_llm.complete.return_value = f'```json\n{json.dumps({"verdict": "upheld", "reasoning": "ok"})}\n```'

        result = a.debate_claim(claim)

        assert result.debate_resolution == "upheld"


class TestFailClosedBehavior:
    def test_malformed_json_response_fails_closed_to_rejected(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.CHALLENGED)
        mock_llm.complete.return_value = "not valid json at all {{{"

        result = a.debate_claim(claim)

        assert result.debate_resolution == "rejected"

    def test_unrecognised_verdict_value_fails_closed_to_rejected(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.CHALLENGED)
        mock_llm.complete.return_value = json.dumps({"verdict": "maybe", "reasoning": "unclear"})

        result = a.debate_claim(claim)

        assert result.debate_resolution == "rejected"

    def test_no_retrieved_evidence_rejects_without_llm_call(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.CHALLENGED, retrieved_evidence="(no evidence retrieved from knowledge base)")

        result = a.debate_claim(claim)

        assert result.debate_resolution == "rejected"
        mock_llm.complete.assert_not_called()

    def test_none_retrieved_evidence_rejects_without_llm_call(self, agent):
        a, mock_llm = agent
        claim = make_claim(GroundingVerdict.CHALLENGED, retrieved_evidence=None)

        result = a.debate_claim(claim)

        assert result.debate_resolution == "rejected"
        mock_llm.complete.assert_not_called()


class TestBatchDebate:
    def test_only_challenged_claims_incur_llm_calls(self, agent):
        a, mock_llm = agent
        mock_llm.complete.return_value = json.dumps({"verdict": "upheld", "reasoning": "ok"})

        claims = [
            make_claim(GroundingVerdict.GROUNDED),
            make_claim(GroundingVerdict.CHALLENGED),
            make_claim(GroundingVerdict.BLOCKED),
            make_claim(GroundingVerdict.CHALLENGED),
        ]

        result = a.debate_claims(claims)

        assert mock_llm.complete.call_count == 2  # only the 2 CHALLENGED claims
        assert result is claims  # same list, mutated in place

    def test_mixed_debate_outcomes_in_a_batch(self, agent):
        a, mock_llm = agent
        mock_llm.complete.side_effect = [
            json.dumps({"verdict": "upheld", "reasoning": "ok"}),
            json.dumps({"verdict": "rejected", "reasoning": "no"}),
        ]

        claims = [make_claim(GroundingVerdict.CHALLENGED), make_claim(GroundingVerdict.CHALLENGED)]
        result = a.debate_claims(claims)

        assert [c.debate_resolution for c in result] == ["upheld", "rejected"]
