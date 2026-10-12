"""
Unit tests for verifyfin.evaluation.run_metrics.

Pure logic plus mocked claims/GroqClient/VectorStore -- runs offline.
"""

from unittest.mock import MagicMock

import pytest

from verifyfin.evaluation.run_metrics import RunMetrics, collect_run_metrics
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, GroundingVerdict, SourceReference


def make_claim(verdict: GroundingVerdict, debate_resolution=None) -> ClaimObject:
    claim = ClaimObject(
        claim_text="A claim.",
        claim_category=ClaimCategory.NUMERICAL,
        produced_by=AgentRole.FUNDAMENTAL,
        source_reference=SourceReference(source_type="edgar_xbrl", document_id="doc1"),
        confidence_score=0.95,
        ticker="AAPL",
        time_period="2026-Q3",
    )
    claim.verdict = verdict
    claim.debate_resolution = debate_resolution
    return claim


class TestHallucinationRate:
    def test_none_when_no_claims(self):
        m = RunMetrics(run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="verify_fin")
        assert m.hallucination_rate() is None

    def test_computed_correctly_with_mixed_usability(self):
        m = RunMetrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="verify_fin",
            total_claims=10, usable_claims_count=7,
        )
        assert m.hallucination_rate() == pytest.approx(0.3)

    def test_zero_when_all_claims_usable(self):
        m = RunMetrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="verify_fin",
            total_claims=5, usable_claims_count=5,
        )
        assert m.hallucination_rate() == 0.0


class TestCollectRunMetrics:
    def test_claim_counts_computed_correctly(self):
        claims = [
            make_claim(GroundingVerdict.GROUNDED),
            make_claim(GroundingVerdict.CHALLENGED, debate_resolution="upheld"),
            make_claim(GroundingVerdict.CHALLENGED, debate_resolution="rejected"),
            make_claim(GroundingVerdict.BLOCKED),
        ]

        m = collect_run_metrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="verify_fin",
            runtime_seconds=1.5, claims=claims,
        )

        assert m.total_claims == 4
        assert m.grounded_count == 1
        assert m.challenged_count == 2
        assert m.blocked_count == 1
        assert m.debate_upheld_count == 1
        assert m.debate_rejected_count == 1
        assert m.usable_claims_count == 2  # 1 grounded + 1 upheld-challenged

    def test_no_claims_leaves_counts_at_zero(self):
        m = collect_run_metrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="single_agent",
            runtime_seconds=0.8, claims=None,
        )
        assert m.total_claims == 0
        assert m.grounded_count == 0

    def test_llm_stats_pulled_from_groq_client(self):
        mock_llm = MagicMock()
        mock_llm.get_stats.return_value = {
            "call_count": 5,
            "cache_hit_count": 2,
            "api_call_count": 3,
            "total_prompt_tokens": 1000,
            "total_completion_tokens": 500,
            "total_tokens": 1500,
        }

        m = collect_run_metrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="verify_fin",
            runtime_seconds=1.0, groq_client=mock_llm,
        )

        assert m.llm_call_count == 5
        assert m.llm_cache_hit_count == 2
        assert m.llm_api_call_count == 3
        assert m.llm_total_tokens == 1500

    def test_no_groq_client_leaves_llm_stats_at_zero(self):
        m = collect_run_metrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="single_agent",
            runtime_seconds=1.0, groq_client=None,
        )
        assert m.llm_call_count == 0

    def test_retrieval_stats_pulled_from_vector_store(self):
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"query_count": 13, "total_query_time_seconds": 0.42}

        m = collect_run_metrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="verify_fin",
            runtime_seconds=1.0, vector_store=mock_store,
        )

        assert m.retrieval_query_count == 13
        assert m.retrieval_total_latency_seconds == 0.42

    def test_no_vector_store_leaves_retrieval_stats_at_zero(self):
        m = collect_run_metrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="single_agent",
            runtime_seconds=1.0, vector_store=None,
        )
        assert m.retrieval_query_count == 0

    def test_runtime_is_rounded_and_stored(self):
        m = collect_run_metrics(
            run_id="r1", ticker="AAPL", time_period="2026-Q3", condition="verify_fin",
            runtime_seconds=12.34567,
        )
        assert m.runtime_seconds == 12.3457
