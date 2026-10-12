"""
Unit tests for verifyfin.agents.synthesis_agent.

GroqClient is mocked -- these tests run offline with zero API calls.
"""

from unittest.mock import MagicMock

import pytest

from verifyfin.agents.synthesis_agent import SynthesisAgent
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, GroundingVerdict, SourceReference
from verifyfin.schemas.state import AgentState


def make_claim(verdict: GroundingVerdict, debate_resolution=None, category=ClaimCategory.NUMERICAL) -> ClaimObject:
    claim = ClaimObject(
        claim_text="AAPL reported revenue of $109,417,000,000.",
        claim_category=category,
        produced_by=AgentRole.FUNDAMENTAL,
        source_reference=SourceReference(source_type="edgar_xbrl", document_id="0000320193-26-000020"),
        confidence_score=0.95,
        ticker="AAPL",
        time_period="2026-Q1",
    )
    claim.verdict = verdict
    claim.debate_resolution = debate_resolution
    return claim


def make_state(claims: list[ClaimObject]) -> AgentState:
    state = AgentState(run_id="test-run", ticker="AAPL", time_period="2026-Q1")
    state.verified_claims = claims
    return state


@pytest.fixture
def agent():
    mock_llm = MagicMock()
    a = SynthesisAgent(llm_client=mock_llm)
    return a, mock_llm


class TestSynthesizeWithUsableClaims:
    def test_grounded_claims_are_included_in_the_prompt(self, agent):
        a, mock_llm = agent
        mock_llm.complete.return_value = "Apple reported strong revenue this quarter."
        state = make_state([make_claim(GroundingVerdict.GROUNDED)])

        report = a.synthesize(state)

        assert report == "Apple reported strong revenue this quarter."
        prompt_used = mock_llm.complete.call_args.kwargs["user_prompt"]
        assert "AAPL reported revenue" in prompt_used

    def test_upheld_challenged_claims_are_included(self, agent):
        a, mock_llm = agent
        mock_llm.complete.return_value = "Report text."
        state = make_state([make_claim(GroundingVerdict.CHALLENGED, debate_resolution="upheld")])

        a.synthesize(state)

        prompt_used = mock_llm.complete.call_args.kwargs["user_prompt"]
        assert "AAPL reported revenue" in prompt_used

    def test_report_is_stripped_of_whitespace(self, agent):
        a, mock_llm = agent
        mock_llm.complete.return_value = "  \n  Report with padding.  \n  "
        state = make_state([make_claim(GroundingVerdict.GROUNDED)])

        report = a.synthesize(state)
        assert report == "Report with padding."


class TestSynthesizeWithNoUsableClaims:
    def test_blocked_claims_excluded_produces_no_claims_message(self, agent):
        a, mock_llm = agent
        state = make_state([make_claim(GroundingVerdict.BLOCKED)])

        report = a.synthesize(state)

        assert "No claims about AAPL" in report
        assert "2026-Q1" in report
        mock_llm.complete.assert_not_called()  # must not waste an LLM call with nothing verified

    def test_rejected_challenged_claims_excluded_produces_no_claims_message(self, agent):
        a, mock_llm = agent
        state = make_state([make_claim(GroundingVerdict.CHALLENGED, debate_resolution="rejected")])

        report = a.synthesize(state)

        assert "No claims about AAPL" in report
        mock_llm.complete.assert_not_called()

    def test_pending_challenged_claims_not_yet_debated_are_excluded(self, agent):
        """A CHALLENGED claim with no debate_resolution yet must NOT be treated as usable."""
        a, mock_llm = agent
        state = make_state([make_claim(GroundingVerdict.CHALLENGED, debate_resolution=None)])

        report = a.synthesize(state)

        assert "No claims about AAPL" in report
        mock_llm.complete.assert_not_called()

    def test_empty_claims_list_produces_no_claims_message(self, agent):
        a, mock_llm = agent
        state = make_state([])

        report = a.synthesize(state)

        assert "No claims about AAPL" in report
        mock_llm.complete.assert_not_called()


class TestMixedClaims:
    def test_only_usable_claims_reach_the_prompt_not_blocked_ones(self, agent):
        a, mock_llm = agent
        mock_llm.complete.return_value = "Report."
        usable = make_claim(GroundingVerdict.GROUNDED)
        blocked = make_claim(GroundingVerdict.BLOCKED)
        blocked.claim_text = "This blocked claim must never appear in the prompt."
        state = make_state([usable, blocked])

        a.synthesize(state)

        prompt_used = mock_llm.complete.call_args.kwargs["user_prompt"]
        assert "must never appear" not in prompt_used


class TestUnverifiedBaselineMode:
    """
    Tests for require_verification=False -- the mode the Unverified
    Multi-Agent baseline uses so it can reuse this exact agent with
    verification as the only variable that differs from VERIFY-FIN.
    """

    def test_uses_raw_claims_regardless_of_verdict(self, agent):
        a, mock_llm = agent
        mock_llm.complete.return_value = "Report."
        state = make_state([])
        # raw_claims includes an UNVERIFIED claim that would normally never pass is_usable_in_final_report()
        unverified_claim = make_claim(GroundingVerdict.UNVERIFIED)
        state.raw_claims = [unverified_claim]

        a.synthesize(state, require_verification=False)

        prompt_used = mock_llm.complete.call_args.kwargs["user_prompt"]
        assert "AAPL reported revenue" in prompt_used

    def test_blocked_claims_are_included_in_baseline_mode(self, agent):
        """The whole point of the baseline: nothing is filtered out, including claims that would be BLOCKED."""
        a, mock_llm = agent
        mock_llm.complete.return_value = "Report."
        state = make_state([])
        state.raw_claims = [make_claim(GroundingVerdict.BLOCKED)]

        a.synthesize(state, require_verification=False)

        mock_llm.complete.assert_called_once()  # must actually attempt synthesis, not skip due to BLOCKED verdict

    def test_empty_raw_claims_still_produces_no_claims_message(self, agent):
        a, mock_llm = agent
        state = make_state([])
        state.raw_claims = []

        report = a.synthesize(state, require_verification=False)

        assert "No claims about AAPL" in report
        mock_llm.complete.assert_not_called()

    def test_default_behavior_unchanged_when_flag_omitted(self, agent):
        """Backward compatibility: omitting the new parameter must behave exactly as before."""
        a, mock_llm = agent
        state = make_state([make_claim(GroundingVerdict.BLOCKED)])

        report = a.synthesize(state)  # no require_verification arg

        assert "No claims about AAPL" in report
        mock_llm.complete.assert_not_called()
