"""Tests for verifyfin.evaluation.fault_injection (pure logic; no knowledge base needed)."""

import random

import pytest

from verifyfin.agents.verification_agent import VerificationAgent
from verifyfin.evaluation import fault_injection as fi

N1 = "Apple reported total revenue of $109,417,000,000 for the period ending June 27, 2026."
EPS = "Apple reported basic earnings per share of $2.03 for the period ending March 28, 2026."
M1 = "On 2026-07-01, the Federal Funds Rate was 3.63."


def base_numeric(text=N1, period_end="2026-06-27"):
    return dict(claim_id="n1", ticker="AAPL", time_period="AAPL-2026-Q2", period_end=period_end, category="numerical",
                text=text, excerpt=fi.reconstruct_numerical_excerpt(text, period_end))


def base_macro(text=M1):
    return dict(claim_id="m1", ticker="AAPL", time_period="AAPL-2026-Q2", period_end="2026-06-27", category="macro",
                text=text, excerpt=fi.reconstruct_macro_excerpt(text))


def variants_by_op(base, **kw):
    vs, drops = fi.generate_variants(base, **kw)
    out = {}
    for v in vs:
        out.setdefault(v["operator"], []).append(v)
    return out, drops


class TestExcerptReconstruction:
    def test_numerical_excerpt_format(self):
        assert fi.reconstruct_numerical_excerpt(N1, "2026-06-27") == "Reconstructed: 109417000000 USD (period N/A to 2026-06-27)"

    def test_per_share_unit(self):
        assert fi.reconstruct_numerical_excerpt(EPS, "2026-03-28") == "Reconstructed: 2.03 USD/shares (period N/A to 2026-03-28)"

    def test_billion_word_is_converted(self):
        ex = fi.reconstruct_numerical_excerpt("Revenue was $109.417 billion.", "2026-06-27")
        assert ex.startswith("Reconstructed: 109417000000 USD")

    def test_macro_excerpt_format(self):
        assert fi.reconstruct_macro_excerpt(M1) == "Reconstructed: 3.63 (as of 2026-07-01)"

    @pytest.mark.parametrize("text,value,date", [
        ("The Federal Funds Rate was 3.64 on 2026-04-01.", "3.64", "2026-04-01"),
        ("On 2026-04-01, U.S. Gross Domestic Product (GDP) was 32563.0.", "32563.0", "2026-04-01"),
        ("As of 2026-01-01, the U.S. unemployment rate was 4.3.", "4.3", "2026-01-01"),
        ("On 2026-07-01, the Consumer Price Index (CPI) was 332.813.", "332.813", "2026-07-01"),
    ])
    def test_macro_value_and_date_are_not_confused(self, text, value, date):
        ex = fi.reconstruct_macro_excerpt(text)
        assert ex == f"Reconstructed: {value} (as of {date})"

    def test_reconstructed_numerical_excerpt_parses_with_the_real_verifier(self):
        ex = fi.reconstruct_numerical_excerpt(N1, "2026-06-27")
        value, unit = VerificationAgent._extract_value_and_unit_from_excerpt(ex)
        assert value == 109417000000.0 and unit == "USD"
        assert VerificationAgent._extract_date_from_excerpt(ex) == "2026-06-27"
        assert "109,417" in VerificationAgent._number_text_variants(value, unit)   # the "in millions" form filings use

    def test_reconstructed_eps_excerpt_parses_with_the_real_verifier(self):
        value, unit = VerificationAgent._extract_value_and_unit_from_excerpt(fi.reconstruct_numerical_excerpt(EPS, "2026-03-28"))
        assert value == 2.03 and unit == "USD/shares"
        assert "2.03" in VerificationAgent._number_text_variants(value, unit)

    def test_reconstructed_macro_excerpt_parses_with_the_real_verifier(self):
        ex = fi.reconstruct_macro_excerpt(M1)
        value, unit = VerificationAgent._extract_value_and_unit_from_excerpt(ex)
        assert value == 3.63 and unit is None
        assert VerificationAgent._extract_date_from_excerpt(ex) == "2026-07-01"

    def test_no_amount_returns_none(self):
        assert fi.reconstruct_numerical_excerpt("Revenue grew strongly.", "2026-06-27") is None


class TestNumericOperators:
    def test_scale_5pct_exact_text(self):
        v = variants_by_op(base_numeric())[0]["scale_5pct"]
        assert {x["mode"] for x in v} == {"phrasing", "source"}
        assert "$114,887,850,000" in v[0]["text"]

    def test_eps_scale_keeps_decimal_style(self):
        v = variants_by_op(base_numeric(EPS, "2026-03-28"))[0]["scale_5pct"][0]
        assert "$2.13" in v["text"]                      # 2.03 * 1.05 = 2.1315

    def test_one_percent_change_is_actually_a_change(self):
        out, _ = variants_by_op(base_numeric(EPS, "2026-03-28"))
        assert "$2.05" in out["scale_1pct"][0]["text"]

    def test_scale_10x(self):
        assert "$1,094,170,000,000" in variants_by_op(base_numeric())[0]["scale_10x"][0]["text"]

    def test_transpose_keeps_digit_multiset_and_changes_value(self):
        t = variants_by_op(base_numeric())[0]["digit_transpose"][0]["text"]
        old, new = fi.parse_amount(N1), fi.parse_amount(t)
        assert new["digits"] != old["digits"] and len(new["digits"]) == len(old["digits"])
        assert sorted(c for c in new["digits"] if c.isdigit()) == sorted(c for c in old["digits"] if c.isdigit())
        assert new["display"] != old["display"] and new["digits"][0] != "0"

    def test_digit_noise_changes_exactly_k_digits(self):
        out, _ = variants_by_op(base_numeric())
        old = fi.parse_amount(N1)["digits"]
        for op, k in (("digit_noise_1", 1), ("digit_noise_3", 3)):
            new = fi.parse_amount(out[op][0]["text"])["digits"]
            assert sum(1 for a, b in zip(old, new) if a != b) == k

    def test_label_swap_changes_only_the_label(self):
        v = variants_by_op(base_numeric())[0]["label_swap"]
        assert len(v) == 1 and v[0]["mode"] == "label"
        assert "net income of $109,417,000,000" in v[0]["text"]
        assert v[0]["meta"]["from_concept"] == "revenue" and v[0]["meta"]["to_concept"] == "net_income"
        assert v[0]["excerpt"] == base_numeric()["excerpt"]        # excerpt carries value+date only, unchanged

    def test_basic_diluted_swap(self):
        v = variants_by_op(base_numeric(EPS, "2026-03-28"))[0]["label_swap"][0]
        assert "diluted earnings per share of $2.03" in v["text"] and v["meta"]["to_concept"] == "eps_diluted"

    def test_period_shift_changes_the_year_and_the_source_excerpt_date(self):
        vs = variants_by_op(base_numeric())[0]["period_shift_year"]
        by_mode = {v["mode"]: v for v in vs}
        assert "June 27, 2025" in by_mode["phrasing"]["text"]
        assert by_mode["phrasing"]["excerpt"] == base_numeric()["excerpt"]                       # phrasing fault: source still true
        assert by_mode["source"]["excerpt"].endswith("to 2025-06-27)")                           # source fault: cited period shifted
        assert by_mode["source"]["period_end"] == "2025-06-27"

    def test_source_mode_excerpt_carries_the_corrupted_value_phrasing_keeps_the_true_one(self):
        by_mode = {v["mode"]: v for v in variants_by_op(base_numeric())[0]["scale_25pct"]}
        assert "136771250000" in by_mode["source"]["excerpt"]
        assert "109417000000" in by_mode["phrasing"]["excerpt"]


class TestMacroOperators:
    def test_scale(self):
        out, _ = variants_by_op(base_macro())
        assert "was 3.81." in out["scale_5pct"][0]["text"]            # 3.63 * 1.05 = 3.8115
        assert "was 4.54." in out["scale_25pct"][0]["text"]

    def test_scale_does_not_touch_the_date(self):
        for v in variants_by_op(base_macro())[0]["scale_25pct"]:
            assert "2026-07-01" in v["text"]

    def test_digit_noise_changes_value_not_date(self):
        v = variants_by_op(base_macro())[0]["digit_noise_1"][0]
        assert "2026-07-01" in v["text"] and fi.find_macro_value(v["text"])["token"] != "3.63"

    def test_date_shifts(self):
        out, _ = variants_by_op(base_macro())
        assert "2026-08-01" in out["date_shift_month"][0]["text"]
        assert "2025-07-01" in out["date_shift_year"][0]["text"]
        assert out["date_shift_month"][0]["meta"] == {"series": "FEDFUNDS", "date": "2026-08-01", "claimed_value": 3.63}

    def test_month_shift_across_year_end(self):
        assert fi._shift_iso("2025-12-01", months=1) == "2026-01-01"
        assert fi._shift_iso("2026-01-31", months=1) == "2026-02-28"

    def test_series_swap_keeps_value_and_date(self):
        v = variants_by_op(base_macro())[0]["series_swap"][0]
        assert v["text"] == "On 2026-07-01, the U.S. unemployment rate was 3.63."
        assert v["meta"] == {"from_series": "FEDFUNDS", "series": "UNRATE", "date": "2026-07-01", "claimed_value": 3.63}

    def test_source_mode_date_shift_moves_the_excerpt_date(self):
        by_mode = {v["mode"]: v for v in variants_by_op(base_macro())[0]["date_shift_month"]}
        assert by_mode["source"]["excerpt"].endswith("(as of 2026-08-01)")
        assert by_mode["phrasing"]["excerpt"].endswith("(as of 2026-07-01)")


class TestValidityHook:
    def test_validator_only_consulted_for_validation_operators(self):
        seen = []
        fi.generate_variants(base_numeric(), validator=lambda b, op, t, m: seen.append(op) or True)
        assert set(seen) == {"label_swap", "period_shift_year"}

    def test_false_verdict_drops_the_variant(self):
        out, drops = variants_by_op(base_numeric(), validator=lambda b, op, t, m: False)
        assert "label_swap" not in out and "period_shift_year" not in out
        assert drops["not_false"] == 2 and "scale_5pct" in out          # unvalidated-by-design operators still produced

    def test_unknown_verdict_drops_the_variant(self):
        out, drops = variants_by_op(base_macro(), validator=lambda b, op, t, m: None)
        assert drops["unvalidated"] == 3 and "date_shift_month" not in out and "series_swap" not in out

    def test_validator_receives_meta_needed_to_judge_falsity(self):
        captured = {}
        fi.generate_variants(base_macro(), validator=lambda b, op, t, m: captured.setdefault(op, m) or True)
        assert captured["date_shift_year"]["date"] == "2025-07-01" and captured["series_swap"]["series"] == "UNRATE"

    def test_no_validator_accepts_everything(self):
        out, drops = variants_by_op(base_numeric())
        assert "label_swap" in out and drops["not_false"] == 0


class TestGeneration:
    def test_deterministic_for_a_seed(self):
        a, _ = fi.generate_variants(base_numeric(), seed=7)
        b, _ = fi.generate_variants(base_numeric(), seed=7)
        assert [v["text"] for v in a] == [v["text"] for v in b]

    def test_different_seed_changes_random_operators(self):
        a = {v["operator"]: v["text"] for v in fi.generate_variants(base_numeric(), seed=1)[0]}
        b = {v["operator"]: v["text"] for v in fi.generate_variants(base_numeric(), seed=2)[0]}
        assert a["scale_5pct"] == b["scale_5pct"]                     # deterministic operators ignore the seed
        assert a["digit_noise_3"] != b["digit_noise_3"]

    def test_every_variant_differs_from_its_original(self):
        for base in (base_numeric(), base_numeric(EPS, "2026-03-28"), base_macro()):
            for v in fi.generate_variants(base)[0]:
                assert v["text"] != base["text"]

    def test_sentiment_claims_are_not_corrupted(self):
        b = dict(base_macro(), category="sentiment")
        assert fi.generate_variants(b)[0] == []

    def test_claim_without_amount_is_inapplicable(self):
        b = base_numeric("Revenue grew strongly.")
        vs, drops = fi.generate_variants(b)
        assert vs == [] and drops["inapplicable"] > 0

    def test_operator_filter(self):
        vs, _ = fi.generate_variants(base_numeric(), operators=["scale_5pct"])
        assert {v["operator"] for v in vs} == {"scale_5pct"}

    def test_variant_counts(self):
        vs, _ = fi.generate_variants(base_numeric())
        assert len(vs) == 8 * 2 + 1       # 8 two-mode operators + 1 label operator
        vm, _ = fi.generate_variants(base_macro())
        assert len(vm) == 6 * 2 + 1


class TestAggregation:
    def rows(self):
        return ([dict(base_id=f"b{i}", verdict="grounded", score=0.9, op="x") for i in range(3)]
                + [dict(base_id=f"b{i}", verdict="challenged", score=0.6, op="x") for i in range(3, 5)]
                + [dict(base_id="b5", verdict="blocked", score=0.3, op="x")])

    def test_summarize_counts_and_rates(self):
        s = fi.summarize(self.rows(), ["op"], n_boot=200)[0]
        assert (s["n"], s["grounded"], s["challenged"], s["blocked"]) == (6, 3, 2, 1)
        assert s["flag_rate"] == pytest.approx(0.5) and s["block_rate"] == pytest.approx(1 / 6)
        assert s["wilson_lo"] < 0.5 < s["wilson_hi"]

    def test_cluster_bootstrap_degenerate_cases(self):
        assert fi.cluster_bootstrap_rate(["a", "b", "c"], [True, True, True]) == (1.0, 1.0)
        assert fi.cluster_bootstrap_rate(["a", "b", "c"], [False, False, False]) == (0.0, 0.0)

    def test_cluster_bootstrap_is_wider_than_wilson_when_variants_are_clustered(self):
        # 10 base claims, 10 variants each; every variant of a base shares the same outcome
        ids = [f"b{i}" for i in range(10) for _ in range(10)]
        flags = [i < 5 for i in range(10) for _ in range(10)]
        lo, hi = fi.cluster_bootstrap_rate(ids, flags, n_boot=3000)
        wl, wh = fi.wilson(sum(flags), len(flags))
        assert (hi - lo) > (wh - wl)

    def test_auroc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score

        pos = [0.9, 0.8, 0.8, 0.7, 0.95]
        neg = [0.6, 0.8, 0.5, 0.4]
        y = [1] * len(pos) + [0] * len(neg)
        assert fi.auroc(pos, neg) == pytest.approx(roc_auc_score(y, pos + neg))

    def test_paired_clean_higher(self):
        r = fi.paired_clean_higher([(0.9, 0.5), (0.8, 0.8), (0.7, 0.9)])
        assert r["higher"] == pytest.approx(1 / 3) and r["equal"] == pytest.approx(1 / 3) and r["lower"] == pytest.approx(1 / 3)
        assert r["mean_drop"] == pytest.approx((0.4 + 0.0 - 0.2) / 3)

    def test_threshold_sweep(self):
        sweep = fi.threshold_sweep([0.9, 0.8, 0.7], [0.5, 0.6, 0.85], [0.6, 0.75, 0.95])
        assert sweep[1] == {"threshold": 0.75, "detection": pytest.approx(2 / 3), "false_flag": pytest.approx(1 / 3), "youden_j": pytest.approx(1 / 3)}
        assert sweep[0]["detection"] == pytest.approx(1 / 3)
