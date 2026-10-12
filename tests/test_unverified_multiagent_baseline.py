"""
Unit tests for verifyfin.baselines.unverified_multiagent_baseline.

All constituent agents are mocked -- these tests run offline with zero
API calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from verifyfin.baselines.unverified_multiagent_baseline import UnverifiedMultiAgentBaseline
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, GroundingVerdict, SourceReference


def make_claim(text="A claim.") -> ClaimObject:
    return ClaimObject(
        claim_text=text,
        claim_category=ClaimCategory.NUMERICAL,
        produced_by=AgentRole.FUNDAMENTAL,
        source_reference=SourceReference(source_type="edgar_xbrl", document_id="doc1"),
        confidence_score=0.95,
        ticker="AAPL",
        time_period="2026-Q3",
    )


@pytest.fixture
def baseline():
    with patch("verifyfin.baselines.unverified_multiagent_baseline.GroqClient") as mock_groq_cls, \
         patch("verifyfin.baselines.unverified_multiagent_baseline.FundamentalAgent") as mock_fund_cls, \
         patch("verifyfin.baselines.unverified_multiagent_baseline.MacroAgent") as mock_macro_cls, \
         patch("verifyfin.baselines.unverified_multiagent_baseline.SentimentAgent") as mock_sent_cls, \
         patch("verifyfin.baselines.unverified_multiagent_baseline.SynthesisAgent") as mock_synth_cls:
        mock_llm = MagicMock()
        mock_llm.get_stats.return_value = {
            "call_count": 0, "cache_hit_count": 0, "api_call_count": 0,
            "total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0,
        }
        mock_groq_cls.return_value = mock_llm

        mock_fund = MagicMock()
        mock_macro = MagicMock()
        mock_sent = MagicMock()
        mock_synth = MagicMock()
        mock_fund_cls.return_value = mock_fund
        mock_macro_cls.return_value = mock_macro
        mock_sent_cls.return_value = mock_sent
        mock_synth_cls.return_value = mock_synth

        b = UnverifiedMultiAgentBaseline()

    return b, mock_fund, mock_macro, mock_sent, mock_synth


class TestUnverifiedMultiAgentBaseline:
    def test_combines_claims_from_all_three_agents(self, baseline):
        b, mock_fund, mock_macro, mock_sent, mock_synth = baseline
        mock_fund.analyze.return_value = [make_claim("Fundamental claim")]
        mock_macro.analyze.return_value = [make_claim("Macro claim")]
        mock_sent.analyze.return_value = [make_claim("Sentiment claim")]
        mock_synth.synthesize.return_value = "Final report."

        state = b.run("AAPL", "2026-Q3")

        assert len(state.raw_claims) == 3

    def test_verified_and_debated_claims_remain_empty(self, baseline):
        """Distinguishing property of this baseline: no verification ever occurs."""
        b, mock_fund, mock_macro, mock_sent, mock_synth = baseline
        mock_fund.analyze.return_value = [make_claim()]
        mock_macro.analyze.return_value = []
        mock_sent.analyze.return_value = []
        mock_synth.synthesize.return_value = "Report."

        state = b.run("AAPL", "2026-Q3")

        assert state.verified_claims == []
        assert state.debated_claims == []

    def test_condition_field_is_set_correctly(self, baseline):
        b, mock_fund, mock_macro, mock_sent, mock_synth = baseline
        mock_fund.analyze.return_value = []
        mock_macro.analyze.return_value = []
        mock_sent.analyze.return_value = []
        mock_synth.synthesize.return_value = "Report."

        state = b.run("AAPL", "2026-Q3")

        assert state.condition == "unverified_multiagent"

    def test_synthesis_called_with_require_verification_false(self, baseline):
        b, mock_fund, mock_macro, mock_sent, mock_synth = baseline
        mock_fund.analyze.return_value = []
        mock_macro.analyze.return_value = []
        mock_sent.analyze.return_value = []
        mock_synth.synthesize.return_value = "Report."

        b.run("AAPL", "2026-Q3")

        call_args = mock_synth.synthesize.call_args
        assert call_args.kwargs.get("require_verification") is False

    def test_one_agent_failing_does_not_break_the_others(self, baseline):
        b, mock_fund, mock_macro, mock_sent, mock_synth = baseline
        mock_fund.analyze.side_effect = Exception("fundamental data unavailable")
        mock_macro.analyze.return_value = [make_claim("Macro claim")]
        mock_sent.analyze.return_value = [make_claim("Sentiment claim")]
        mock_synth.synthesize.return_value = "Report."

        state = b.run("AAPL", "2026-Q3")  # must not raise

        assert len(state.raw_claims) == 2  # macro + sentiment still produced
        assert any("Fundamental Agent failed" in e for e in state.error_log)

    def test_final_report_is_set(self, baseline):
        b, mock_fund, mock_macro, mock_sent, mock_synth = baseline
        mock_fund.analyze.return_value = [make_claim()]
        mock_macro.analyze.return_value = []
        mock_sent.analyze.return_value = []
        mock_synth.synthesize.return_value = "The final unverified report text."

        state = b.run("AAPL", "2026-Q3")

        assert state.final_report == "The final unverified report text."

    def test_last_run_metrics_reflects_unverified_claims(self, baseline):
        """
        All raw_claims here stay at verdict=UNVERIFIED (nothing was
        checked), so usable_claims_count must be 0 even though
        total_claims is not -- this is the whole point of the baseline,
        not a bug.
        """
        b, mock_fund, mock_macro, mock_sent, mock_synth = baseline
        mock_fund.analyze.return_value = [make_claim("A"), make_claim("B")]
        mock_macro.analyze.return_value = []
        mock_sent.analyze.return_value = []
        mock_synth.synthesize.return_value = "Report."

        b.run("AAPL", "2026-Q3")

        assert b.last_run_metrics is not None
        assert b.last_run_metrics.condition == "unverified_multiagent"
        assert b.last_run_metrics.total_claims == 2
        assert b.last_run_metrics.usable_claims_count == 0  # nothing was verified
