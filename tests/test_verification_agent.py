"""
Unit tests for verifyfin.agents.verification_agent.

VectorStore is mocked -- these tests run offline, with no real
embedding model or ChromaDB involved.
"""

from unittest.mock import MagicMock

import pytest

from verifyfin.agents.verification_agent import VerificationAgent
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, GroundingVerdict, SourceReference


def make_claim(claim_text: str = "AAPL reported revenue of $95B.", excerpt: str = None) -> ClaimObject:
    return ClaimObject(
        claim_text=claim_text,
        claim_category=ClaimCategory.NUMERICAL,
        produced_by=AgentRole.FUNDAMENTAL,
        source_reference=SourceReference(
            source_type="edgar_xbrl",
            document_id="0000320193-26-000013",
            excerpt=excerpt,
        ),
        confidence_score=0.95,
        ticker="AAPL",
        time_period="2026-Q1",
    )


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unused-for-this-test")
    mock_store = MagicMock()
    a = VerificationAgent(vector_store=mock_store)
    return a, mock_store


class TestQueryTextSelection:
    def test_always_queries_with_claim_text_not_structured_excerpt(self, agent):
        a, mock_store = agent
        claim = make_claim(
            claim_text="Apple reported revenue of $95 billion.",
            excerpt="RevenueFromContract...: 95000000000 USD",
        )
        mock_store.query.return_value = [{"text": "match", "metadata": {}, "similarity_score": 0.9}]

        a.verify_claim(claim)

        query_used = mock_store.query.call_args.args[0]
        assert query_used == "Apple reported revenue of $95 billion."

    def test_uses_claim_text_when_no_excerpt_present(self, agent):
        a, mock_store = agent
        claim = make_claim(claim_text="Revenue grew a lot.", excerpt=None)
        mock_store.query.return_value = [{"text": "match", "metadata": {}, "similarity_score": 0.9}]

        a.verify_claim(claim)

        query_used = mock_store.query.call_args.args[0]
        assert query_used == "Revenue grew a lot."


class TestGroundingRouting:
    def test_high_similarity_routes_to_grounded(self, agent):
        a, mock_store = agent
        claim = make_claim()
        mock_store.query.return_value = [
            {"text": "Apple reported revenue of $95 billion.", "metadata": {}, "similarity_score": 0.88}
        ]

        result = a.verify_claim(claim)

        assert result.verdict == GroundingVerdict.GROUNDED
        assert result.grounding_score == 0.88
        assert result.is_usable_in_final_report() is True

    def test_mid_similarity_routes_to_challenged(self, agent):
        a, mock_store = agent
        claim = make_claim()
        mock_store.query.return_value = [{"text": "Some loosely related text.", "metadata": {}, "similarity_score": 0.60}]

        result = a.verify_claim(claim)

        assert result.verdict == GroundingVerdict.CHALLENGED
        assert result.is_usable_in_final_report() is False

    def test_low_similarity_routes_to_blocked(self, agent):
        a, mock_store = agent
        claim = make_claim()
        mock_store.query.return_value = [{"text": "Unrelated text.", "metadata": {}, "similarity_score": 0.2}]

        result = a.verify_claim(claim)

        assert result.verdict == GroundingVerdict.BLOCKED
        assert result.is_usable_in_final_report() is False

    def test_uses_best_match_when_multiple_results_returned(self, agent):
        a, mock_store = agent
        claim = make_claim()
        mock_store.query.return_value = [
            {"text": "best match", "metadata": {}, "similarity_score": 0.9},
            {"text": "worse match", "metadata": {}, "similarity_score": 0.5},
        ]

        result = a.verify_claim(claim)

        assert result.grounding_score == 0.9
        assert result.retrieved_evidence == "best match"


class TestEmptyKnowledgeBase:
    def test_no_results_routes_to_blocked_not_error(self, agent):
        a, mock_store = agent
        claim = make_claim()
        mock_store.query.return_value = []

        result = a.verify_claim(claim)

        assert result.verdict == GroundingVerdict.BLOCKED
        assert result.grounding_score == 0.0
        assert result.retrieved_evidence is not None


class TestBatchVerification:
    def test_verifies_all_claims_and_returns_same_list(self, agent):
        a, mock_store = agent
        claims = [make_claim("Claim A"), make_claim("Claim B")]
        mock_store.query.return_value = [{"text": "match", "metadata": {}, "similarity_score": 0.9}]

        result = a.verify_claims(claims)

        assert len(result) == 2
        assert all(c.verdict == GroundingVerdict.GROUNDED for c in result)
        assert result is claims

    def test_filters_query_by_claims_ticker(self, agent):
        a, mock_store = agent
        claim = make_claim()
        mock_store.query.return_value = [{"text": "match", "metadata": {}, "similarity_score": 0.9}]

        a.verify_claim(claim)

        assert mock_store.query.call_args.kwargs["ticker"] == "AAPL"

    def test_mixed_verdicts_across_a_batch(self, agent):
        a, mock_store = agent
        claims = [make_claim(f"Claim {i}") for i in range(3)]
        scores = [0.9, 0.6, 0.1]
        mock_store.query.side_effect = [
            [{"text": "match", "metadata": {}, "similarity_score": s}] for s in scores
        ]

        result = a.verify_claims(claims)

        assert [c.verdict for c in result] == [
            GroundingVerdict.GROUNDED,
            GroundingVerdict.CHALLENGED,
            GroundingVerdict.BLOCKED,
        ]


class TestThresholdsFromConfig:
    def test_thresholds_loaded_from_config(self, agent):
        a, _ = agent
        assert a.grounded_threshold == 0.75
        assert a.challenged_threshold == 0.55

    def test_boundary_value_at_grounded_threshold_is_grounded(self, agent):
        a, mock_store = agent
        claim = make_claim()
        mock_store.query.return_value = [{"text": "match", "metadata": {}, "similarity_score": a.grounded_threshold}]
        result = a.verify_claim(claim)
        assert result.verdict == GroundingVerdict.GROUNDED

    def test_boundary_value_at_challenged_threshold_is_challenged(self, agent):
        a, mock_store = agent
        claim = make_claim()
        mock_store.query.return_value = [{"text": "match", "metadata": {}, "similarity_score": a.challenged_threshold}]
        result = a.verify_claim(claim)
        assert result.verdict == GroundingVerdict.CHALLENGED


class TestNumberVariantGeneration:
    def test_large_usd_value_includes_millions_representation(self, agent):
        a, _ = agent
        variants = a._number_text_variants(254940000000.0, "USD")
        assert "254,940" in variants
        assert "254,940,000,000" in variants

    def test_small_eps_value_has_no_millions_conversion(self, agent):
        a, _ = agent
        variants = a._number_text_variants(4.87, "USD/shares")
        assert "4.87" in variants
        assert not any("," in v and len(v) > 6 for v in variants)

    def test_none_value_returns_empty_set(self, agent):
        a, _ = agent
        assert a._number_text_variants(None, "USD") == set()

    def test_none_unit_still_produces_variants(self, agent):
        """Regression: FRED-style values (no unit) must still generate usable variants."""
        a, _ = agent
        variants = a._number_text_variants(3.63, None)
        assert "3.63" in variants


class TestExcerptParsing:
    def test_parses_standard_xbrl_excerpt_format(self, agent):
        a, _ = agent
        value, unit = a._extract_value_and_unit_from_excerpt(
            "RevenueFromContractWithCustomerExcludingAssessedTax: 254940000000 USD (period end 2026-03-28)"
        )
        assert value == 254940000000.0
        assert unit == "USD"

    def test_none_excerpt_returns_none_none(self, agent):
        a, _ = agent
        assert a._extract_value_and_unit_from_excerpt(None) == (None, None)

    def test_non_numeric_excerpt_returns_none_none(self, agent):
        a, _ = agent
        assert a._extract_value_and_unit_from_excerpt("currentPrice: not-a-number") == (None, None)

    def test_regression_fred_style_excerpt_with_no_unit_token_now_parses(self, agent):
        """
        Regression test for a real bug found during development: FRED
        excerpts have the format "SERIES: value (as of date)" with NO
        unit token, unlike XBRL's "TAG: value UNIT (...)". The original
        regex required a unit token immediately after the value,
        silently failing to parse every FRED excerpt and disabling
        numeric matching for all macro claims.
        """
        a, _ = agent
        value, unit = a._extract_value_and_unit_from_excerpt("FEDFUNDS: 3.63 (as of 2026-07-01)")
        assert value == 3.63
        assert unit is None  # no unit token present, and that's fine now


class TestHybridNumericVerification:
    def test_numeric_match_is_preferred_over_higher_ranked_semantic_decoy(self, agent):
        a, mock_store = agent
        claim = make_claim(
            claim_text="Apple reported total revenue of $254,940,000,000 for the quarter.",
            excerpt="RevenueFromContractWithCustomerExcludingAssessedTax: 254940000000 USD (period end 2026-03-28)",
        )
        mock_store.query.return_value = [
            {"text": "Date: May 1, 2026 | Apple Inc.", "metadata": {}, "similarity_score": 0.60},
            {"text": "Total net sales | 254,940 | 230,901", "metadata": {}, "similarity_score": 0.50},
        ]

        result = a.verify_claim(claim)

        assert result.retrieved_evidence == "Total net sales | 254,940 | 230,901"
        assert result.grounding_score == 0.75
        assert result.verdict == GroundingVerdict.GROUNDED
        assert mock_store.query.call_count == 1

    def test_correctly_rejects_misleading_row_with_different_number(self, agent):
        a, mock_store = agent
        claim = make_claim(
            claim_text="Apple reported total liabilities of $264,591,000,000.",
            excerpt="Liabilities: 264591000000 USD (period end 2026-03-28)",
        )
        mock_store.query.return_value = [
            {
                "text": "Total liabilities and shareholders' equity | $ | 371,082 | $ | 359,241",
                "metadata": {},
                "similarity_score": 0.58,
            },
            {"text": "Total liabilities | $ | 264,591 | $ | 253,000", "metadata": {}, "similarity_score": 0.40},
        ]

        result = a.verify_claim(claim)

        assert result.retrieved_evidence == "Total liabilities | $ | 264,591 | $ | 253,000"
        assert "371,082" not in result.retrieved_evidence

    def test_targeted_search_finds_match_outside_initial_candidates(self, agent):
        a, mock_store = agent
        claim = make_claim(
            claim_text="Apple reported total revenue of $254,940,000,000 for the quarter.",
            excerpt="RevenueFromContractWithCustomerExcludingAssessedTax: 254940000000 USD (period end 2026-03-28)",
        )

        def query_side_effect(query_text, ticker=None, top_k=8, where_document_contains=None):
            if where_document_contains is not None:
                if where_document_contains in "Total net sales | 254,940 | 230,901":
                    return [{"text": "Total net sales | 254,940 | 230,901", "metadata": {}, "similarity_score": 0.68}]
                return []
            return [
                {"text": "Date: May 1, 2026 | Apple Inc.", "metadata": {}, "similarity_score": 0.60},
                {"text": "Unrelated exhibit index text.", "metadata": {}, "similarity_score": 0.55},
            ]

        mock_store.query.side_effect = query_side_effect

        result = a.verify_claim(claim)

        assert result.retrieved_evidence == "Total net sales | 254,940 | 230,901"
        assert result.grounding_score == min(1.0, 0.68 + 0.25)
        assert mock_store.query.call_count > 1

    def test_falls_back_to_semantic_only_when_number_found_nowhere_in_kb(self, agent):
        a, mock_store = agent
        claim = make_claim(
            claim_text="Apple reported total revenue of $254,940,000,000.",
            excerpt="Revenues: 254940000000 USD (period end 2026-03-28)",
        )

        def query_side_effect(query_text, ticker=None, top_k=8, where_document_contains=None):
            if where_document_contains is not None:
                return []
            return [
                {"text": "No matching number anywhere here.", "metadata": {}, "similarity_score": 0.6},
                {"text": "Also no match.", "metadata": {}, "similarity_score": 0.5},
            ]

        mock_store.query.side_effect = query_side_effect

        result = a.verify_claim(claim)

        assert result.retrieved_evidence == "No matching number anywhere here."
        assert result.grounding_score == 0.6

    def test_falls_back_to_semantic_only_when_excerpt_has_no_parseable_number(self, agent):
        a, mock_store = agent
        claim = make_claim(claim_text="Market sentiment towards AAPL is positive.", excerpt="news_source: Reuters")
        mock_store.query.return_value = [{"text": "Sentiment was positive.", "metadata": {}, "similarity_score": 0.8}]

        result = a.verify_claim(claim)

        assert result.grounding_score == 0.8

    def test_boosted_score_is_capped_at_one(self, agent):
        a, mock_store = agent
        claim = make_claim(excerpt="Revenues: 100 USD (period end 2026-03-28)")
        mock_store.query.return_value = [{"text": "Revenue was 100.", "metadata": {}, "similarity_score": 0.9}]

        result = a.verify_claim(claim)

        assert result.grounding_score == 1.0

    def test_regression_fred_claim_numeric_match_now_works(self, agent):
        """
        Regression test proving the excerpt-parsing fix actually enables
        numeric matching end-to-end for FRED-sourced macro claims, which
        was previously silently disabled entirely.
        """
        a, mock_store = agent
        claim = make_claim(
            claim_text="The Federal Funds Rate was 3.63% as of July 2026.",
            excerpt="FEDFUNDS: 3.63 (as of 2026-07-01)",
        )
        mock_store.query.return_value = [
            {"text": "Unrelated macro commentary.", "metadata": {}, "similarity_score": 0.5},
            {"text": "FEDFUNDS observation: 3.63 on 2026-07-01", "metadata": {}, "similarity_score": 0.45},
        ]

        result = a.verify_claim(claim)

        assert result.retrieved_evidence == "FEDFUNDS observation: 3.63 on 2026-07-01"
        assert result.grounding_score == min(1.0, 0.45 + 0.25)


# ---------------------------------------------------------------------------
# Ablation switches (keyword-only; defaults must reproduce production behaviour)
# ---------------------------------------------------------------------------

def _fred(text, sim):
    return {"text": text, "metadata": {"source_type": "fred"}, "similarity_score": sim}


def _filing(text, sim):
    return {"text": text, "metadata": {"source_type": "edgar_10q"}, "similarity_score": sim}


def _agent_with(results, **flags):
    mock_store = MagicMock()
    mock_store.query.return_value = results
    return VerificationAgent(vector_store=mock_store, **flags), mock_store


class TestAblationSwitches:
    def test_defaults_match_production_behaviour(self):
        a, _ = _agent_with([_filing("x", 0.5)])
        assert (a.use_numeric_match, a.use_targeted_search, a.use_date_preference, a.require_date_match, a.numeric_boost) == (
            True, True, True, False, 0.25)

    def test_semantic_only_disables_the_numeric_boost(self):
        claim = make_claim("Apple reported total revenue of $254,940,000,000.",
                           excerpt="Revenues: 254940000000 USD (period N/A to 2026-03-28)")
        results = [_filing("Total net sales | 254,940 | 230,901", 0.50)]
        full, _ = _agent_with(results)
        sem, _ = _agent_with(results, use_numeric_match=False)
        assert full.verify_claim(claim).grounding_score == pytest.approx(0.75)
        claim2 = make_claim("Apple reported total revenue of $254,940,000,000.",
                            excerpt="Revenues: 254940000000 USD (period N/A to 2026-03-28)")
        assert sem.verify_claim(claim2).grounding_score == pytest.approx(0.50)

    def test_numeric_boost_zero_keeps_the_matched_chunk_but_no_boost(self):
        claim = make_claim("Revenue was $100.", excerpt="Revenues: 100 USD (period N/A to 2026-03-28)")
        a, _ = _agent_with([_filing("a", 0.9), _filing("Revenue | 100", 0.40)], numeric_boost=0.0)
        out = a.verify_claim(claim)
        assert out.retrieved_evidence == "Revenue | 100" and out.grounding_score == pytest.approx(0.40)

    def test_targeted_search_can_be_disabled(self):
        claim = make_claim("Revenue was $254,940,000,000.", excerpt="Revenues: 254940000000 USD (period N/A to 2026-03-28)")
        mock_store = MagicMock()

        def query(text, ticker=None, top_k=8, where_document_contains=None):
            if where_document_contains is not None:
                return [_filing("Total net sales | 254,940", 0.68)]
            return [_filing("cover page", 0.60)]

        mock_store.query.side_effect = query
        on = VerificationAgent(vector_store=mock_store)
        off = VerificationAgent(vector_store=mock_store, use_targeted_search=False)
        assert on.verify_claim(claim).grounding_score == pytest.approx(0.93)
        claim2 = make_claim("Revenue was $254,940,000,000.", excerpt="Revenues: 254940000000 USD (period N/A to 2026-03-28)")
        assert off.verify_claim(claim2).grounding_score == pytest.approx(0.60)

    def test_date_preference_can_be_disabled(self):
        results = [_fred("UNRATE observation: 4.1 on 2026-08-01", 0.80), _fred("UNRATE observation: 4.1 on 2026-07-01", 0.60)]
        c1 = make_claim("On 2026-07-01, the unemployment rate was 4.1.", excerpt="UNRATE: 4.1 (as of 2026-07-01)")
        c2 = make_claim("On 2026-07-01, the unemployment rate was 4.1.", excerpt="UNRATE: 4.1 (as of 2026-07-01)")
        with_pref, _ = _agent_with(results)
        without, _ = _agent_with(results, use_date_preference=False)
        assert with_pref.verify_claim(c1).retrieved_evidence.endswith("2026-07-01")     # prefers the right month
        assert without.verify_claim(c2).retrieved_evidence.endswith("2026-08-01")       # first value match, wrong month


class TestStrictDateRequirement:
    """The wrong-month case from the first evaluation run: a flat series shares its value across months."""

    def test_default_accepts_a_value_match_from_the_wrong_month(self):
        claim = make_claim("On 2026-09-01, the unemployment rate was 4.1.", excerpt="UNRATE: 4.1 (as of 2026-09-01)")
        a, _ = _agent_with([_fred("UNRATE observation: 4.1 on 2026-08-01", 0.70)])
        out = a.verify_claim(claim)
        assert out.grounding_score == pytest.approx(0.95)          # boosted despite the wrong date -> a miss

    def test_strict_mode_refuses_the_wrong_month(self):
        claim = make_claim("On 2026-09-01, the unemployment rate was 4.1.", excerpt="UNRATE: 4.1 (as of 2026-09-01)")
        a, _ = _agent_with([_fred("UNRATE observation: 4.1 on 2026-08-01", 0.70)], require_date_match=True)
        out = a.verify_claim(claim)
        assert out.grounding_score == pytest.approx(0.70)          # no boost: falls back to semantic-only
        assert out.verdict == GroundingVerdict.CHALLENGED

    def test_strict_mode_still_accepts_the_right_month(self):
        claim = make_claim("On 2026-08-01, the unemployment rate was 4.1.", excerpt="UNRATE: 4.1 (as of 2026-08-01)")
        a, _ = _agent_with([_fred("UNRATE observation: 4.1 on 2026-08-01", 0.70)], require_date_match=True)
        assert a.verify_claim(claim).grounding_score == pytest.approx(0.95)

    def test_strict_mode_does_not_date_constrain_filing_chunks(self):
        """Filings write 'March 28, 2026', never ISO dates, so requiring an ISO date there would reject every filing claim."""
        claim = make_claim("Revenue was $254,940,000,000.", excerpt="Revenues: 254940000000 USD (period N/A to 2026-03-28)")
        a, _ = _agent_with([_filing("Total net sales | 254,940 | 230,901 (March 28, 2026)", 0.50)], require_date_match=True)
        assert a.verify_claim(claim).grounding_score == pytest.approx(0.75)

    def test_prefer_date_qualified_handles_missing_metadata(self):
        out = VerificationAgent._prefer_date_qualified([{"text": "x 4.1", "similarity_score": 0.5}], "2026-01-01", require=True)
        assert out is not None  # no metadata -> treated as non-FRED, so not date-constrained


class TestTextConsistencyCheck:
    """v2: the number/date in the SENTENCE must agree with the cited excerpt (off by default)."""

    EXCERPT = "Revenues: 116017000000 USD (period 2026-03-29 to 2026-06-27)"
    CLEAN = "Apple reported total revenue of $116,017,000,000 for the period ended June 27, 2026."
    WRONG_NUMBER = "Apple reported total revenue of $161,017,000,000 for the period ended June 27, 2026."
    WRONG_YEAR = "Apple reported total revenue of $116,017,000,000 for the period ended June 27, 2025."
    EVIDENCE = [{"text": "Total net sales 116,017 for the three months ended June 27, 2026", "metadata": {}, "similarity_score": 0.62}]

    def _agent(self, monkeypatch, **kw):
        monkeypatch.setenv("GROQ_API_KEY", "unused")
        store = MagicMock()
        store.query.return_value = self.EVIDENCE
        return VerificationAgent(vector_store=store, **kw)

    def test_off_by_default_wrong_sentence_still_passes(self, monkeypatch):
        a = self._agent(monkeypatch)
        c = a.verify_claim(make_claim(self.WRONG_NUMBER, self.EXCERPT))
        assert c.verdict == GroundingVerdict.GROUNDED          # the documented v1 blind spot

    def test_on_clean_claim_unchanged(self, monkeypatch):
        a = self._agent(monkeypatch, check_text_consistency=True)
        assert a.verify_claim(make_claim(self.CLEAN, self.EXCERPT)).verdict == GroundingVerdict.GROUNDED

    def test_on_wrong_number_is_challenged(self, monkeypatch):
        a = self._agent(monkeypatch, check_text_consistency=True)
        c = a.verify_claim(make_claim(self.WRONG_NUMBER, self.EXCERPT))
        assert c.verdict == GroundingVerdict.CHALLENGED
        assert c.grounding_score < a.grounded_threshold

    def test_on_wrong_year_is_challenged(self, monkeypatch):
        a = self._agent(monkeypatch, check_text_consistency=True)
        assert a.verify_claim(make_claim(self.WRONG_YEAR, self.EXCERPT)).verdict == GroundingVerdict.CHALLENGED

    def test_block_action_sends_mismatch_to_blocked(self, monkeypatch):
        a = self._agent(monkeypatch, check_text_consistency=True, text_mismatch_action="block")
        assert a.verify_claim(make_claim(self.WRONG_NUMBER, self.EXCERPT)).verdict == GroundingVerdict.BLOCKED

    def test_invalid_action_rejected(self, monkeypatch):
        with pytest.raises(ValueError):
            self._agent(monkeypatch, text_mismatch_action="ignore")

    def test_counts_are_recorded(self, monkeypatch):
        a = self._agent(monkeypatch, check_text_consistency=True)
        a.verify_claim(make_claim(self.CLEAN, self.EXCERPT))
        a.verify_claim(make_claim(self.WRONG_NUMBER, self.EXCERPT))
        assert a.text_check_counts["ok"] == 1 and a.text_check_counts["mismatch"] == 1

    def test_claim_without_excerpt_is_not_penalised(self, monkeypatch):
        a = self._agent(monkeypatch, check_text_consistency=True)
        c = a.verify_claim(make_claim("Revenue grew a lot.", None))
        assert c.grounding_score == pytest.approx(0.62)


# ---------------------------------------------------------------------------------------------------------------
# v3 evidence alignment
# ---------------------------------------------------------------------------------------------------------------

# Real chunks copied from the AAPL-2026-Q2 knowledge base (also used in test_evidence_alignment.py).
CH_SALES = ("Three Months Ended | Nine Months Ended | June 27,2026 | June 28,2025 | Change | June 27,2026 | June 28,2025 | Change || "
            "Total net sales | $ | 109,417 | $ | 94,036 | 16% | $ | 364,357 | $ | 313,695 | 16%")
CH_TA = "June 27,2026 | September 27,2025 | ASSETS: | Current assets: || Total assets | $ | 383,266 | $ | 359,241"


class ChunkStore:
    """Deterministic store: chunks are (text, source_type, similarity). `where_document_contains` is honoured."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.contains_calls = []

    def query(self, query_text, ticker=None, top_k=5, where_document_contains=None):
        self.contains_calls.append(where_document_contains)
        out = [{"text": t, "metadata": {"source_type": st, "ticker": "AAPL"}, "similarity_score": sim}
               for t, st, sim in self.chunks if not where_document_contains or where_document_contains in t]
        return sorted(out, key=lambda r: -r["similarity_score"])[:top_k]


def fred_claim(text, excerpt):
    return ClaimObject(claim_text=text, claim_category=ClaimCategory.MACRO, produced_by=AgentRole.MACRO,
                       source_reference=SourceReference(source_type="fred", document_id="FEDFUNDS", excerpt=excerpt),
                       confidence_score=0.95, ticker="AAPL", time_period="AAPL-2026-Q1")


def filing_claim(text, excerpt):
    return ClaimObject(claim_text=text, claim_category=ClaimCategory.NUMERICAL, produced_by=AgentRole.FUNDAMENTAL,
                       source_reference=SourceReference(source_type="edgar_xbrl", document_id="0000320193-26-000020", excerpt=excerpt),
                       confidence_score=0.95, ticker="AAPL", time_period="AAPL-2026-Q2")


V3 = dict(date_aligned_search=True, label_check=True, filing_period_check=True)
FFR_TEXT = "The Federal Funds Rate was 3.64 on 2026-04-01."
FFR_EXCERPT = "FEDFUNDS: 3.64 (as of 2026-04-01)"


def ffr_store():
    """The REAL failure: the 03-01 chunk (same value, wrong month) is in the top-8 pool, the 04-01 chunk is not."""
    distractors = [(f"Headline number {i}: apple supply chain update", "news", 0.85 - 0.01 * i) for i in range(7)]
    return ChunkStore([("FEDFUNDS observation: 3.64 on 2026-03-01", "fred", 0.90),
                       ("FEDFUNDS observation: 3.64 on 2026-04-01", "fred", 0.40), *distractors])


class TestEvidenceAlignmentFred:
    def test_v1_grounds_the_claim_to_the_wrong_month(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        c = fred_claim(FFR_TEXT, FFR_EXCERPT)
        VerificationAgent(vector_store=ffr_store()).verify_claim(c)
        assert c.verdict == GroundingVerdict.GROUNDED
        assert "2026-03-01" in c.retrieved_evidence                      # right number, wrong date: the documented failure

    def test_date_aligned_search_cites_the_april_record(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        a = VerificationAgent(vector_store=ffr_store(), date_aligned_search=True)
        c = fred_claim(FFR_TEXT, FFR_EXCERPT)
        a.verify_claim(c)
        assert c.retrieved_evidence == "FEDFUNDS observation: 3.64 on 2026-04-01"
        assert a.alignment_counts["record_found"] == 1
        assert c.verdict == GroundingVerdict.CHALLENGED                 # similarity of the right chunk is only 0.40+0.25

    def test_exact_record_grounds_lifts_a_confirmed_record_to_grounded(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        a = VerificationAgent(vector_store=ffr_store(), exact_record_grounds=True, **V3)
        c = fred_claim(FFR_TEXT, FFR_EXCERPT)
        a.verify_claim(c)
        assert c.verdict == GroundingVerdict.GROUNDED and c.retrieved_evidence.endswith("on 2026-04-01")
        assert a.alignment_counts["exact_lift"] == 1

    def test_a_date_with_no_such_record_cannot_be_grounded(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        text, excerpt = "The Federal Funds Rate was 3.64 on 2025-04-01.", "FEDFUNDS: 3.64 (as of 2025-04-01)"
        v1, v3 = fred_claim(text, excerpt), fred_claim(text, excerpt)
        VerificationAgent(vector_store=ffr_store()).verify_claim(v1)
        a = VerificationAgent(vector_store=ffr_store(), date_aligned_search=True)
        a.verify_claim(v3)
        assert v1.verdict == GroundingVerdict.GROUNDED                  # v1 accepts the value in another month
        assert v3.verdict == GroundingVerdict.CHALLENGED and a.alignment_counts["record_missing"] == 1

    def test_series_swap_is_caught_by_the_label_check(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        text = "The U.S. unemployment rate was 3.64 on 2026-04-01."      # value and date belong to FEDFUNDS
        a = VerificationAgent(vector_store=ffr_store(), exact_record_grounds=True, **V3)
        c = fred_claim(text, FFR_EXCERPT)
        a.verify_claim(c)
        assert c.verdict != GroundingVerdict.GROUNDED and a.alignment_counts["label_mismatch"] == 1
        assert a.alignment_counts["exact_lift"] == 0                     # a mismatch is never lifted

    def test_exact_record_does_not_override_the_text_check(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        a = VerificationAgent(vector_store=ffr_store(), exact_record_grounds=True, check_text_consistency=True, **V3)
        c = fred_claim("The Federal Funds Rate was 9.99 on 2026-04-01.", FFR_EXCERPT)   # sentence disagrees with the record
        a.verify_claim(c)
        assert c.verdict == GroundingVerdict.CHALLENGED

    def test_no_flags_means_exactly_the_old_behaviour_and_no_extra_queries(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        store = ffr_store()
        VerificationAgent(vector_store=store).verify_claim(fred_claim(FFR_TEXT, FFR_EXCERPT))
        assert store.contains_calls == [None]                             # the date-substring search was not issued


class TestEvidenceAlignmentFilings:
    def store(self, extra=()):
        return ChunkStore([(CH_SALES, "edgar_10q", 0.70), (CH_TA, "edgar_10q", 0.68), *extra])

    def test_label_swap_with_the_right_number(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        text, excerpt = "Apple's total liabilities were $383,266,000,000.", "Liabilities: 383266000000 USD (period 2025-12-28 to 2026-06-27)"
        v1, v3 = filing_claim(text, excerpt), filing_claim(text, excerpt)
        VerificationAgent(vector_store=self.store()).verify_claim(v1)
        a = VerificationAgent(vector_store=self.store(), label_check=True)
        a.verify_claim(v3)
        assert v1.verdict == GroundingVerdict.GROUNDED                    # 383,266 is found (under Total assets)
        assert v3.verdict == GroundingVerdict.CHALLENGED and a.alignment_counts["label_mismatch"] == 1

    def test_correct_filing_claim_is_untouched_by_v3(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        text, excerpt = "Apple reported total revenue of $109,417,000,000.", "Revenues: 109417000000 USD (period 2026-03-29 to 2026-06-27)"
        a = VerificationAgent(vector_store=self.store(), exact_record_grounds=True, **V3)
        c = filing_claim(text, excerpt)
        a.verify_claim(c)
        assert c.verdict == GroundingVerdict.GROUNDED
        assert a.alignment_counts["label_mismatch"] == 0 and a.alignment_counts["period_mismatch"] == 0

    def test_wrong_period_column_is_flagged(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        text = "Apple reported total revenue of $109,417,000,000 for the period ending June 28, 2025."
        excerpt = "Revenues: 109417000000 USD (period 2025-03-30 to 2025-06-28)"        # current value, prior-year date
        v1, v3 = filing_claim(text, excerpt), filing_claim(text, excerpt)
        VerificationAgent(vector_store=self.store()).verify_claim(v1)
        a = VerificationAgent(vector_store=self.store(), filing_period_check=True)
        a.verify_claim(v3)
        assert v1.verdict == GroundingVerdict.GROUNDED
        assert v3.verdict == GroundingVerdict.CHALLENGED and a.alignment_counts["period_mismatch"] == 1

    def test_another_chunk_in_the_right_column_is_preferred_to_a_flag(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        other = ("June 27,2026 | June 28,2025 || Total net sales | $ | 100 | $ | 109,417", "edgar_10q", 0.60)
        text = "Apple reported total revenue of $109,417,000,000 for the period ending June 28, 2025."
        excerpt = "Revenues: 109417000000 USD (period 2025-03-30 to 2025-06-28)"
        a = VerificationAgent(vector_store=self.store([other]), filing_period_check=True, label_check=True)
        c = filing_claim(text, excerpt)
        a.verify_claim(c)
        assert a.alignment_counts["period_reselected"] == 1 and c.retrieved_evidence == other[0]
        assert c.verdict == GroundingVerdict.GROUNDED

    def test_unalignable_chunk_is_never_penalised(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "x")
        store = ChunkStore([("Total net sales were 109,417 in the quarter (prose, no table header)", "edgar_10q", 0.70)])
        text, excerpt = "Apple reported total revenue of $109,417,000,000.", "Revenues: 109417000000 USD (period 2026-03-29 to 2026-06-27)"
        a = VerificationAgent(vector_store=store, **V3)
        c = filing_claim(text, excerpt)
        a.verify_claim(c)
        assert c.verdict == GroundingVerdict.GROUNDED and a.alignment_counts["period_mismatch"] == 0
