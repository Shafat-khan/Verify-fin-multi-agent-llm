"""Tests for fault_validators: corruptions are kept only if PROVABLY false."""

import pytest

from verifyfin.evaluation import fault_injection as fi
from verifyfin.evaluation import fault_validators as fv


class FredStore:
    """Minimal knowledge-base stand-in holding FRED observation chunks."""
    def __init__(self, obs):
        self.obs = obs  # {(series, date): value}

    def query(self, text, ticker=None, top_k=5, where_document_contains=None):
        out = []
        for (s, d), v in self.obs.items():
            chunk = f"{s} observation: {v} on {d}"
            if where_document_contains is None or where_document_contains in chunk:
                out.append({"text": chunk, "metadata": {"source_type": "fred"}, "similarity_score": 0.5})
        return out[:top_k]


OBS = {("UNRATE", "2026-07-01"): 4.1, ("UNRATE", "2026-08-01"): 4.1, ("UNRATE", "2025-07-01"): 4.3,
       ("FEDFUNDS", "2026-07-01"): 3.63, ("FEDFUNDS", "2026-08-01"): 3.80, ("FEDFUNDS", "2025-07-01"): 4.33,
       ("CPIAUCSL", "2026-07-01"): 332.813}
TRUTH = {"revenue": {"value": 109417000000}, "net_income": {"value": 29789000000}, "eps_basic": {"value": 2.03},
         "eps_diluted": {"value": 2.03}, "total_assets": {"value": 383266000000}, "total_liabilities": {"value": 275746000000}}


def validator(known=("2026-06-27", "2026-03-28", "2025-12-27"), truth_by_period=None):
    if truth_by_period is None:
        truth_by_period = {"2026-06-27": TRUTH}
    return fv.make_validator(ticker="AAPL", period_end="2026-06-27", store=FredStore(OBS),
                             xbrl_truth=lambda t, pe: truth_by_period.get(pe), known_period_ends=list(known))


class TestHelpers:
    def test_clearly_different_thresholds(self):
        assert fv.clearly_different(4.5, 4.1) and not fv.clearly_different(4.1, 4.1)
        assert not fv.clearly_different(4.12, 4.1)             # within 0.05 absolute: ambiguous
        assert not fv.clearly_different(32800, 32563.0)        # within 1% relative: ambiguous
        assert fv.clearly_different(120000, 109417)

    def test_fred_value_reads_the_chunk(self):
        s = FredStore(OBS)
        assert fv.fred_value(s, "AAPL", "FEDFUNDS", "2026-07-01") == 3.63
        assert fv.fred_value(s, "AAPL", "UNRATE", "2026-08-01") == 4.1
        assert fv.fred_value(s, "AAPL", "UNRATE", "2030-01-01") is None

    def test_fred_value_survives_a_failing_store(self):
        class Boom:
            def query(self, *a, **k): raise RuntimeError("x")
        assert fv.fred_value(Boom(), "AAPL", "UNRATE", "2026-07-01") is None


class TestMacroValidation:
    def test_flat_series_date_shift_is_rejected_as_not_false(self):
        """Unemployment was 4.1% in both July and August: shifting the date does NOT make the claim false."""
        v = validator()
        meta = {"series": "UNRATE", "date": "2026-08-01", "claimed_value": 4.1}
        assert v({}, "date_shift_month", "x", meta) is False

    def test_moving_series_date_shift_is_accepted(self):
        v = validator()
        assert v({}, "date_shift_month", "x", {"series": "FEDFUNDS", "date": "2026-08-01", "claimed_value": 3.63}) is True

    def test_year_shift(self):
        v = validator()
        assert v({}, "date_shift_year", "x", {"series": "FEDFUNDS", "date": "2025-07-01", "claimed_value": 3.63}) is True

    def test_series_swap_accepted_when_values_differ(self):
        v = validator()
        assert v({}, "series_swap", "x", {"series": "UNRATE", "date": "2026-07-01", "claimed_value": 3.63}) is True

    def test_unknown_when_the_date_is_not_in_the_knowledge_base(self):
        v = validator()
        assert v({}, "date_shift_month", "x", {"series": "FEDFUNDS", "date": "2031-01-01", "claimed_value": 3.63}) is None


class TestNumericValidation:
    def test_label_swap_accepted_when_values_differ(self):
        v = validator()
        meta = {"from_concept": "revenue", "to_concept": "net_income", "claimed_value": 109417000000}
        assert v({}, "label_swap", "x", meta) is True

    def test_basic_diluted_swap_rejected_when_eps_are_equal(self):
        """Basic and diluted EPS are both 2.03 here: swapping the label leaves a TRUE claim."""
        v = validator()
        meta = {"from_concept": "eps_basic", "to_concept": "eps_diluted", "claimed_value": 2.03}
        assert v({}, "label_swap", "x", meta) is False

    def test_label_swap_unknown_without_xbrl_truth(self):
        v = validator(truth_by_period={})
        assert v({}, "label_swap", "x", {"to_concept": "net_income", "claimed_value": 1.0}) is None

    def test_period_shift_accepted_when_value_appears_in_no_other_period(self):
        v = validator(truth_by_period={"2026-06-27": TRUTH, "2026-03-28": {"revenue": {"value": 95359000000}},
                                       "2025-12-27": {"revenue": {"value": 143756000000}}})
        assert v({}, "period_shift_year", "x", {"concept": "revenue", "claimed_value": 109417000000}) is True

    def test_period_shift_rejected_when_value_repeats_in_another_period(self):
        v = validator(truth_by_period={"2026-06-27": TRUTH, "2026-03-28": {"revenue": {"value": 109417000000}}})
        assert v({}, "period_shift_year", "x", {"concept": "revenue", "claimed_value": 109417000000}) is False

    def test_unrelated_operator_returns_none(self):
        assert validator()({}, "scale_5pct", "x", {}) is None


class TestIntegrationWithGenerateVariants:
    def test_only_provably_false_variants_survive(self):
        base = dict(claim_id="m", ticker="AAPL", time_period="T", period_end="2026-06-27", category="macro",
                    text="On 2026-07-01, the U.S. unemployment rate was 4.1.", excerpt="Reconstructed: 4.1 (as of 2026-07-01)")
        vs, drops = fi.generate_variants(base, validator=validator())
        ops = {v["operator"] for v in vs}
        assert "date_shift_month" not in ops        # UNRATE is flat 4.1 in July/August -> would be a TRUE claim
        assert "date_shift_year" in ops and "series_swap" in ops
        assert drops["not_false"] == 1
