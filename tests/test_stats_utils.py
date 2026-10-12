"""
Tests for verifyfin.evaluation.stats_utils, cross-checked against
scipy / scikit-learn where a reference implementation exists. Includes
the exact scenario that exposed the two metric errors found during the
cross-verification audit.
"""

import math

import pytest

from verifyfin.evaluation.stats_utils import (
    cohens_kappa,
    mcnemar_exact,
    mean_sd_ci,
    paired_summary,
    rankdata_average,
    signed_rank_components,
    wilcoxon_signflip,
    wilson_interval,
)


class TestRanks:
    def test_average_ranks_with_ties(self):
        assert rankdata_average([0.1, 0.1, 0.3]) == [1.5, 1.5, 3.0]

    def test_matches_scipy_rankdata(self):
        from scipy import stats

        vals = [0.5, 0.2, 0.2, 0.9, 0.5, 0.5]
        assert rankdata_average(vals) == list(stats.rankdata(vals))


class TestWilcoxonSignFlip:
    def test_all_positive_n3_is_one_eighth(self):
        r = wilcoxon_signflip([1, 2, 3], "greater")
        assert r["p_value"] == pytest.approx(0.125)
        assert r["rank_biserial"] == pytest.approx(1.0)

    def test_matches_scipy_exact_when_no_ties(self):
        from scipy import stats

        d = [1.0, -2.0, 3.0, 4.5, -0.7, 2.2, 1.1, 0.4]
        mine = wilcoxon_signflip(d, "greater")["p_value"]
        ref = stats.wilcoxon(d, alternative="greater", method="exact").pvalue
        assert mine == pytest.approx(ref, abs=1e-12)

    def test_two_sided_matches_scipy_exact(self):
        from scipy import stats

        d = [1.0, -2.0, 3.0, 4.5, -0.7, 2.2, 1.1, 0.4]
        mine = wilcoxon_signflip(d, "two-sided")["p_value"]
        ref = stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue
        assert mine == pytest.approx(ref, abs=1e-12)

    def test_zeros_are_dropped(self):
        a = wilcoxon_signflip([0, 0, 1, 2, 3, 0], "greater")
        b = wilcoxon_signflip([1, 2, 3], "greater")
        assert a["p_value"] == b["p_value"] and a["n_nonzero"] == 3

    def test_regression_user_scenario_exact_p_is_0_375_not_0_2071(self):
        """
        14 scenarios: 11 ties, 2 where VERIFY-FIN lower, 1 higher, tied ranks.
        scipy's default normal approximation gave p = 0.2071; the exact
        permutation p-value is 0.375.
        """
        from scipy import stats

        d = [0.08, -0.08, 0.10] + [0.0] * 11
        approx_p = stats.wilcoxon(d, alternative="greater").pvalue
        exact = wilcoxon_signflip(d, "greater")
        assert approx_p == pytest.approx(0.2071, abs=1e-4)
        assert exact["p_value"] == pytest.approx(0.375)
        assert exact["t_plus"] == pytest.approx(4.5) and exact["t_minus"] == pytest.approx(1.5)

    def test_no_nonzero_differences_gives_p_one(self):
        assert wilcoxon_signflip([0, 0, 0])["p_value"] == 1.0


class TestEffectSizes:
    def test_rank_biserial_uses_rank_sums_not_sign_counts(self):
        """The previously printed 0.333 was (n+ - n-)/n; the rank-biserial is 0.5."""
        comp = signed_rank_components([0.08, -0.08, 0.10] + [0.0] * 11)
        assert comp["sign_net_advantage"] == pytest.approx(1 / 3)
        assert comp["rank_biserial"] == pytest.approx(0.5)

    def test_rank_biserial_bounds(self):
        assert signed_rank_components([1, 2, 3])["rank_biserial"] == pytest.approx(1.0)
        assert signed_rank_components([-1, -2, -3])["rank_biserial"] == pytest.approx(-1.0)


class TestMeanSdCi:
    def test_matches_numpy_and_scipy(self):
        import numpy as np
        from scipy import stats

        v = [0.0769, 0.0833, 0.0, 0.2222, 0.0833, 0.1538]
        r = mean_sd_ci(v)
        assert r["mean"] == pytest.approx(np.mean(v))
        assert r["sd"] == pytest.approx(np.std(v, ddof=1))
        assert r["se"] == pytest.approx(stats.sem(v))
        lo, hi = stats.t.interval(0.95, len(v) - 1, loc=np.mean(v), scale=stats.sem(v))
        assert r["lo"] == pytest.approx(lo) and r["hi"] == pytest.approx(hi)

    def test_user_summary_reconciles_sd_se_and_ci(self):
        """mean 8.50, SD 5.42, n=14 -> SE 1.449 -> CI half-width 3.13 -> [5.37, 11.63]."""
        from scipy import stats

        se = 5.42 / math.sqrt(14)
        half = stats.t.ppf(0.975, 13) * se
        assert se == pytest.approx(1.449, abs=1e-3)
        assert (8.50 - half, 8.50 + half) == (pytest.approx(5.37, abs=0.01), pytest.approx(11.63, abs=0.01))

    def test_nan_values_are_ignored(self):
        assert mean_sd_ci([1.0, float("nan"), 3.0])["n"] == 2


class TestPairedSummary:
    def test_sign_convention_positive_means_second_is_lower(self):
        baseline = [0.10, 0.20, 0.30]
        verify_fin = [0.05, 0.20, 0.35]
        r = paired_summary(baseline, verify_fin, n_boot=200)
        assert r["diffs"] == [pytest.approx(0.05), pytest.approx(0.0), pytest.approx(-0.05)]
        assert (r["n_positive"], r["n_negative"], r["n_tied"]) == (1, 1, 1)

    def test_bootstrap_is_deterministic(self):
        a, b = [0.1, 0.2, 0.3, 0.4], [0.05, 0.2, 0.1, 0.3]
        assert paired_summary(a, b, n_boot=300)["boot_lo"] == paired_summary(a, b, n_boot=300)["boot_lo"]


class TestWilsonKappaMcNemar:
    def test_wilson_matches_scipy(self):
        from scipy import stats

        for k, n in [(0, 10), (5, 20), (154, 157), (9, 12), (3, 157)]:
            ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
            lo, hi = wilson_interval(k, n)
            assert lo == pytest.approx(ref.low, abs=1e-9) and hi == pytest.approx(ref.high, abs=1e-9)

    def test_kappa_matches_sklearn(self):
        from sklearn.metrics import cohen_kappa_score

        tp, fp, fn, tn = 154, 3, 5, 9
        human = [1] * (tp + fn) + [0] * (fp + tn)
        system = [1] * tp + [0] * fn + [1] * fp + [0] * tn
        assert cohens_kappa(tp, fp, fn, tn) == pytest.approx(cohen_kappa_score(human, system), abs=1e-12)
        assert cohens_kappa(tp, fp, fn, tn) == pytest.approx(0.667, abs=1e-3)

    def test_mcnemar_matches_binomial_test(self):
        from scipy import stats

        assert mcnemar_exact(9, 5) == pytest.approx(stats.binomtest(9, 14, 0.5).pvalue, abs=1e-12)
        assert mcnemar_exact(9, 5) == pytest.approx(0.424, abs=1e-3)
        assert mcnemar_exact(0, 0) == 1.0
