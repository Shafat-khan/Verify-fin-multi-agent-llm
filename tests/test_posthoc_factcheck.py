"""
Unit tests for verifyfin.evaluation.posthoc_factcheck.

All external dependencies (GroqClient, VectorStore, VerificationAgent,
DebateAgent) are mocked -- runs offline.
"""

from unittest.mock import MagicMock, patch

import pytest

from verifyfin.evaluation.posthoc_factcheck import (
    _repair_truncated_claim_array,
    _build_claim_objects,
    extract_claims_from_text,
    factcheck_single_agent_report,
    factcheck_unverified_baseline_claims,
)
from verifyfin.evaluation.run_metrics import RunMetrics
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, GroundingVerdict, SourceReference


def make_claim(verdict=GroundingVerdict.UNVERIFIED, debate_resolution=None) -> ClaimObject:
    c = ClaimObject(
        claim_text="A claim.",
        claim_category=ClaimCategory.NUMERICAL,
        produced_by=AgentRole.FUNDAMENTAL,
        source_reference=SourceReference(source_type="edgar_xbrl", document_id="doc1"),
        confidence_score=0.9,
        ticker="AAPL",
        time_period="2026-Q3",
    )
    c.verdict = verdict
    c.debate_resolution = debate_resolution
    return c


def make_metrics() -> RunMetrics:
    return RunMetrics(run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="single_agent")


class TestExtractClaimsFromText:
    def test_parses_valid_json_array(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '["Revenue was $10B.", "EPS was $2.00."]'

        claims = extract_claims_from_text(mock_llm, "Some report text.")

        assert claims == ["Revenue was $10B.", "EPS was $2.00."]

    def test_parses_markdown_fenced_json(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '```json\n["Claim A"]\n```'

        claims = extract_claims_from_text(mock_llm, "Some report text.")

        assert claims == ["Claim A"]

    def test_empty_report_text_returns_empty_without_llm_call(self):
        mock_llm = MagicMock()

        claims = extract_claims_from_text(mock_llm, "")

        assert claims == []
        mock_llm.complete.assert_not_called()

    def test_malformed_json_returns_empty_list_not_error(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "not json at all"

        claims = extract_claims_from_text(mock_llm, "Some report text.")

        assert claims == []

    def test_non_string_items_are_filtered_out(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '["Real claim", 42, null, "Another claim"]'

        claims = extract_claims_from_text(mock_llm, "Some report text.")

        assert claims == ["Real claim", "Another claim"]

    def test_non_list_json_returns_empty(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '{"not": "a list"}'

        claims = extract_claims_from_text(mock_llm, "Some report text.")

        assert claims == []

    def test_truncated_response_recovers_complete_claims(self):
        """
        Regression test for a real bug: a response cut off mid-string
        (hitting the token limit) used to discard the ENTIRE array
        rather than recovering the claims that were actually complete
        before the cutoff.
        """
        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '["Apple posted first-quarter revenue of roughly $124 billion", '
            '"iPhone shipments rose 9% in the quarter", '
            '"Apple Services revenue grew 12% to $22 billio'  # cut off mid-string
        )

        claims = extract_claims_from_text(mock_llm, "Some report text.")

        assert claims == [
            "Apple posted first-quarter revenue of roughly $124 billion",
            "iPhone shipments rose 9% in the quarter",
        ]

    def test_extraction_call_uses_higher_max_tokens(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '["Claim A"]'

        extract_claims_from_text(mock_llm, "Some report text.")

        assert mock_llm.complete.call_args.kwargs["max_tokens"] == 3000


class TestRepairTruncatedClaimArray:
    def test_recovers_all_complete_strings(self):
        text = '["Claim one", "Claim two", "Claim thre'
        assert _repair_truncated_claim_array(text) == ["Claim one", "Claim two"]

    def test_empty_text_returns_empty_list(self):
        assert _repair_truncated_claim_array("") == []

    def test_no_complete_strings_returns_empty_list(self):
        assert _repair_truncated_claim_array('["incomplete cut off mid') == []

    def test_trivially_short_matches_are_filtered_as_noise(self):
        text = '["", "ab", "A genuinely long and complete claim statement"]'
        result = _repair_truncated_claim_array(text)
        assert result == ["A genuinely long and complete claim statement"]


class TestBuildClaimObjects:
    def test_builds_one_claim_object_per_text(self):
        objs = _build_claim_objects(["Claim A", "Claim B"], "AAPL", "2026-Q3")

        assert len(objs) == 2
        assert objs[0].claim_text == "Claim A"
        assert objs[1].claim_text == "Claim B"

    def test_claims_have_no_source_excerpt(self):
        """The Single-Agent baseline never cited anything -- this must be reflected honestly."""
        objs = _build_claim_objects(["Claim A"], "AAPL", "2026-Q3")

        assert objs[0].source_reference.excerpt is None

    def test_ticker_and_period_are_set_correctly(self):
        objs = _build_claim_objects(["Claim A"], "aapl", "2026-Q3")

        assert objs[0].ticker == "AAPL"
        assert objs[0].time_period == "2026-Q3"


class TestFactcheckUnverifiedBaselineClaims:
    def test_empty_raw_claims_sets_zero_without_calling_agents(self):
        metrics = make_metrics()
        mock_store = MagicMock()
        mock_llm = MagicMock()

        with patch("verifyfin.evaluation.posthoc_factcheck.VerificationAgent") as mock_va_cls:
            factcheck_unverified_baseline_claims([], mock_store, mock_llm, metrics)

        assert metrics.posthoc_total_claims == 0
        mock_va_cls.assert_not_called()

    def test_runs_verification_and_debate_and_populates_metrics(self):
        metrics = make_metrics()
        mock_store = MagicMock()
        mock_llm = MagicMock()
        raw_claims = [make_claim(), make_claim()]

        with patch("verifyfin.evaluation.posthoc_factcheck.VerificationAgent") as mock_va_cls, \
             patch("verifyfin.evaluation.posthoc_factcheck.DebateAgent") as mock_da_cls:
            mock_va = MagicMock()
            mock_da = MagicMock()
            mock_va_cls.return_value = mock_va
            mock_da_cls.return_value = mock_da

            verified = [make_claim(GroundingVerdict.GROUNDED), make_claim(GroundingVerdict.CHALLENGED)]
            debated = [make_claim(GroundingVerdict.GROUNDED), make_claim(GroundingVerdict.CHALLENGED, "upheld")]
            mock_va.verify_claims.return_value = verified
            mock_da.debate_claims.return_value = debated

            factcheck_unverified_baseline_claims(raw_claims, mock_store, mock_llm, metrics)

        assert metrics.posthoc_total_claims == 2
        assert metrics.posthoc_usable_count == 2  # 1 grounded + 1 upheld-challenged
        mock_va.verify_claims.assert_called_once()
        mock_da.debate_claims.assert_called_once_with(verified)


class TestFactcheckSingleAgentReport:
    def test_no_extractable_claims_sets_failure_flag(self):
        metrics = make_metrics()
        mock_store = MagicMock()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "not valid json"

        factcheck_single_agent_report("Some report.", "AAPL", "2026-Q3", mock_store, mock_llm, metrics)

        assert metrics.posthoc_total_claims == 0
        assert metrics.posthoc_extraction_failed is True

    def test_extracted_claims_are_fact_checked(self):
        metrics = make_metrics()
        mock_store = MagicMock()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '["Apple revenue was $94B."]'

        with patch("verifyfin.evaluation.posthoc_factcheck.VerificationAgent") as mock_va_cls, \
             patch("verifyfin.evaluation.posthoc_factcheck.DebateAgent") as mock_da_cls:
            mock_va = MagicMock()
            mock_da = MagicMock()
            mock_va_cls.return_value = mock_va
            mock_da_cls.return_value = mock_da

            blocked_claim = make_claim(GroundingVerdict.BLOCKED)
            mock_va.verify_claims.return_value = [blocked_claim]
            mock_da.debate_claims.return_value = [blocked_claim]

            factcheck_single_agent_report("Apple revenue was $94B.", "AAPL", "2026-Q3", mock_store, mock_llm, metrics)

        assert metrics.posthoc_total_claims == 1
        assert metrics.posthoc_usable_count == 0
        assert metrics.posthoc_extraction_failed is False


class TestPosthocHallucinationRate:
    def test_none_when_no_posthoc_data(self):
        metrics = make_metrics()
        assert metrics.posthoc_hallucination_rate() is None

    def test_computed_correctly(self):
        metrics = make_metrics()
        metrics.posthoc_total_claims = 10
        metrics.posthoc_usable_count = 3
        assert metrics.posthoc_hallucination_rate() == pytest.approx(0.7)

    def test_zero_total_claims_returns_none_not_divide_by_zero(self):
        metrics = make_metrics()
        metrics.posthoc_total_claims = 0
        assert metrics.posthoc_hallucination_rate() is None


class TestReturnValues:
    """The fact-checked claims are now returned so callers can persist them (e.g. for human labelling)."""

    def test_single_agent_returns_the_fact_checked_claims(self):
        metrics = make_metrics()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = '["Claim one is long enough.", "Claim two is long enough."]'
        with patch("verifyfin.evaluation.posthoc_factcheck.VerificationAgent") as va_cls, \
             patch("verifyfin.evaluation.posthoc_factcheck.DebateAgent") as da_cls:
            checked = [make_claim(GroundingVerdict.GROUNDED), make_claim(GroundingVerdict.BLOCKED)]
            va_cls.return_value.verify_claims.return_value = checked
            da_cls.return_value.debate_claims.return_value = checked
            out = factcheck_single_agent_report("report", "AAPL", "2026-Q3", MagicMock(), mock_llm, metrics)
        assert out == checked and metrics.posthoc_total_claims == 2

    def test_single_agent_returns_empty_list_when_nothing_extracted(self):
        metrics = make_metrics()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "not json"
        assert factcheck_single_agent_report("report", "AAPL", "2026-Q3", MagicMock(), mock_llm, metrics) == []

    def test_unverified_returns_claims_or_empty_list(self):
        metrics = make_metrics()
        assert factcheck_unverified_baseline_claims([], MagicMock(), MagicMock(), metrics) == []
        with patch("verifyfin.evaluation.posthoc_factcheck.VerificationAgent") as va_cls, \
             patch("verifyfin.evaluation.posthoc_factcheck.DebateAgent") as da_cls:
            checked = [make_claim(GroundingVerdict.GROUNDED)]
            va_cls.return_value.verify_claims.return_value = checked
            da_cls.return_value.debate_claims.return_value = checked
            assert factcheck_unverified_baseline_claims([make_claim()], MagicMock(), MagicMock(), make_metrics()) == checked
