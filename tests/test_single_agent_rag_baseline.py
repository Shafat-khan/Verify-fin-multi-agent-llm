"""Tests for the Single-Agent + RAG baseline (offline: fake LLM and fake knowledge base)."""

from unittest.mock import MagicMock

import pytest

from verifyfin.baselines.single_agent_rag_baseline import RETRIEVAL_QUERIES, SingleAgentRagBaseline


class FakeStore:
    def __init__(self):
        self.queries = []
        self.stats_reset = False

    def query(self, text, ticker=None, top_k=5, where_document_contains=None):
        self.queries.append((text, ticker, top_k))
        base = f"passage for [{text[:12]}]"
        # every query also returns one shared passage, to exercise de-duplication
        return [{"text": base, "metadata": {"source_type": "edgar_10q"}, "similarity_score": 0.8},
                {"text": "SHARED passage", "metadata": {"source_type": "fred"}, "similarity_score": 0.7}][:top_k]

    def reset_stats(self):
        self.stats_reset = True

    def get_stats(self):
        return {"query_count": len(self.queries), "total_query_time_seconds": 0.01}


@pytest.fixture
def baseline():
    llm = MagicMock()
    llm.complete.return_value = "  A grounded report.  "
    llm.get_stats.return_value = {"call_count": 1, "cache_hit_count": 0, "api_call_count": 1,
                                  "total_prompt_tokens": 3000, "total_completion_tokens": 300, "total_tokens": 3300}
    return SingleAgentRagBaseline(llm_client=llm), llm


class TestRetrieval:
    def test_uses_the_same_fixed_queries_for_every_company_and_filters_by_ticker(self, baseline):
        b, _ = baseline
        s = FakeStore()
        b.retrieve_context(s, "AAPL")
        assert [q for q, _, _ in s.queries] == RETRIEVAL_QUERIES
        assert all(t == "AAPL" and k == 2 for _, t, k in s.queries)

    def test_passages_are_deduplicated_preserving_order(self, baseline):
        b, _ = baseline
        ctx = b.retrieve_context(FakeStore(), "AAPL")
        texts = [p["text"] for p in ctx]
        assert texts.count("SHARED passage") == 1 and len(texts) == len(RETRIEVAL_QUERIES) + 1
        assert texts[0].startswith("passage for")


class TestPrompt:
    def test_prompt_contains_evidence_and_forbids_outside_knowledge(self, baseline):
        b, llm = baseline
        b.run("aapl", "AAPL-2026-Q2", FakeStore(), period_end="2026-06-27")
        kw = llm.complete.call_args.kwargs
        assert "ONLY the numbered evidence passages" in kw["system_prompt"]
        assert "Do not use any outside knowledge" in kw["system_prompt"]
        assert "Company: AAPL" in kw["user_prompt"] and "(period ended 2026-06-27)" in kw["user_prompt"]
        assert "[1] (edgar_10q) passage for" in kw["user_prompt"] and "(fred) SHARED passage" in kw["user_prompt"]

    def test_long_passages_are_truncated(self):
        llm = MagicMock()
        llm.complete.return_value = "r"
        llm.get_stats.return_value = {"call_count": 1, "cache_hit_count": 0, "api_call_count": 1,
                                      "total_prompt_tokens": 1, "total_completion_tokens": 1, "total_tokens": 2}
        b = SingleAgentRagBaseline(llm_client=llm, max_passage_chars=50)

        class Long(FakeStore):
            def query(self, text, ticker=None, top_k=5, where_document_contains=None):
                return [{"text": "x" * 500, "metadata": {"source_type": "edgar_10q"}}]

        b.run("AAPL", "T", Long())
        assert "x" * 51 not in llm.complete.call_args.kwargs["user_prompt"]

    def test_no_structured_claims_or_verification_are_requested(self, baseline):
        b, llm = baseline
        b.run("AAPL", "T", FakeStore())
        text = llm.complete.call_args.kwargs["system_prompt"].lower()
        assert "json" not in text and "claim" not in text


class TestRunAndMetrics:
    def test_returns_stripped_report_and_records_condition(self, baseline):
        b, _ = baseline
        s = FakeStore()
        report = b.run("AAPL", "AAPL-2026-Q2", s)
        assert report == "A grounded report."
        m = b.last_run_metrics
        assert m.condition == "single_agent_rag" and m.ticker == "AAPL" and m.llm_total_tokens == 3300
        assert m.retrieval_query_count == len(RETRIEVAL_QUERIES)
        assert s.stats_reset                                   # per-run retrieval counters

    def test_context_is_kept_for_inspection(self, baseline):
        b, _ = baseline
        b.run("AAPL", "T", FakeStore())
        assert len(b.last_context) == len(RETRIEVAL_QUERIES) + 1

    def test_exactly_one_llm_call(self, baseline):
        b, llm = baseline
        b.run("AAPL", "T", FakeStore())
        llm.complete.assert_called_once()


class TestEmptyReportHandling:
    """The first real run returned an EMPTY report for 4/14 scenarios (reasoning tokens exhausted max_tokens=700)."""

    def test_retries_with_a_larger_budget_when_the_first_reply_is_empty(self):
        from verifyfin.baselines.single_agent_rag_baseline import _MAX_TOKENS, _RETRY_MAX_TOKENS
        llm = MagicMock()
        llm.complete.side_effect = ["", "A real report."]
        llm.get_stats.return_value = {"call_count": 2, "cache_hit_count": 0, "api_call_count": 2,
                                      "total_prompt_tokens": 1, "total_completion_tokens": 1, "total_tokens": 2}
        b = SingleAgentRagBaseline(llm_client=llm)
        assert b.run("AAPL", "2026-Q2", FakeStore()) == "A real report."
        budgets = [c.kwargs["max_tokens"] for c in llm.complete.call_args_list]
        assert budgets == [_MAX_TOKENS, _RETRY_MAX_TOKENS] and _MAX_TOKENS > 700
        assert b.last_report_empty is False and b.last_attempts == 2

    def test_still_empty_is_flagged_not_hidden(self):
        llm = MagicMock()
        llm.complete.return_value = None          # content=None is what an exhausted reasoning model can return
        llm.get_stats.return_value = {"call_count": 2, "cache_hit_count": 0, "api_call_count": 2,
                                      "total_prompt_tokens": 1, "total_completion_tokens": 1, "total_tokens": 2}
        b = SingleAgentRagBaseline(llm_client=llm)
        assert b.run("AAPL", "2026-Q2", FakeStore()) == ""
        assert b.last_report_empty is True

    def test_first_attempt_success_makes_one_call(self, baseline):
        b, llm = baseline
        b.run("AAPL", "2026-Q2", FakeStore())
        assert llm.complete.call_count == 1 and b.last_attempts == 1
