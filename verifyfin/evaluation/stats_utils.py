"""
verifyfin.evaluation.stats_utils
===================================
Small, dependency-light statistical helpers shared by
scripts/analyze_results.py and scripts/audit_metrics.py.

Why this module exists (found while cross-verifying the reported
metrics):
    * scipy.stats.wilcoxon's DEFAULT method falls back to a normal
      approximation when the sample has zero differences or tied ranks.
      With only a handful of non-zero paired differences (this project:
      3 of 14 scenarios) that approximation is badly anti-conservative.
      It reported p = 0.2071 where the exact permutation p-value is
      0.375. wilcoxon_signflip() computes the EXACT conditional p-value
      (valid with ties and zeros) by dynamic programming over the
      doubled ranks.
    * The "rank-biserial correlation" was previously computed from SIGN
      COUNTS, (n+ - n-)/n, which is the sign-test net advantage and not
      the rank-biserial correlation. The rank-biserial correlation
      (Kerby, 2014) uses RANK SUMS: r = (T+ - T-)/(T+ + T-). Both are
      returned here under their correct names.

Sign convention used throughout the project for paired comparisons:
    difference = rate(baseline) - rate(VERIFY-FIN)
so a POSITIVE difference means VERIFY-FIN has the LOWER unsupported rate.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence


def _clean(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if v is not None and v == v]


def mean_sd_ci(values: Sequence[float], confidence: float = 0.95) -> dict:
    """
    Mean, sample SD (ddof=1), standard error of the mean, and a
    t-distribution confidence interval. NOTE the three different
    "plus/minus" quantities this distinguishes:
        mean +/- SD   : spread of the scenarios themselves
        mean +/- SE   : precision of the mean (SD / sqrt(n))
        mean +/- t*SE : the 95% confidence interval half-width
    """
    from scipy import stats

    v = _clean(values)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = sum(v) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": float("nan"), "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))
    se = sd / math.sqrt(n)
    half = float(stats.t.ppf((1 + confidence) / 2, n - 1)) * se
    return {"n": n, "mean": mean, "sd": sd, "se": se, "lo": mean - half, "hi": mean + half, "half_width": half}


def rankdata_average(values: Sequence[float], tol: float = 1e-12) -> list[float]:
    """Average ranks (1-based) with ties; ties detected within an absolute tolerance."""
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and abs(values[idx[j + 1]] - values[idx[i]]) <= tol:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def signed_rank_components(diffs: Sequence[float]) -> dict:
    """T+, T-, counts and the two (correctly named) effect sizes."""
    nz = [d for d in _clean(diffs) if d != 0]
    m = len(nz)
    n_pos = sum(1 for d in nz if d > 0)
    n_neg = m - n_pos
    if m == 0:
        return {"n_nonzero": 0, "n_pos": 0, "n_neg": 0, "t_plus": 0.0, "t_minus": 0.0,
                "rank_biserial": float("nan"), "sign_net_advantage": float("nan")}
    ranks = rankdata_average([abs(d) for d in nz])
    t_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    t_minus = sum(r for r, d in zip(ranks, nz) if d < 0)
    return {
        "n_nonzero": m, "n_pos": n_pos, "n_neg": n_neg, "t_plus": t_plus, "t_minus": t_minus,
        # Kerby (2014): the simple-difference formula, from RANK SUMS
        "rank_biserial": (t_plus - t_minus) / (t_plus + t_minus),
        # sign-test analogue: share of non-tied pairs favouring one side minus the other
        "sign_net_advantage": (n_pos - n_neg) / m,
    }


def wilcoxon_signflip(diffs: Sequence[float], alternative: str = "greater") -> dict:
    """
    EXACT Wilcoxon signed-rank test by sign-flip permutation, conditional
    on the observed (tie-averaged) ranks. Zeros are dropped (Wilcoxon's
    method). Valid with ties. alternative: 'greater' (T+ large), 'less',
    or 'two-sided'.

    Implementation: ranks are multiples of 0.5, so doubling them gives
    integers; a subset-sum DP then counts, over all 2^m equally likely
    sign assignments, how many give each possible T+.
    """
    comp = signed_rank_components(diffs)
    m = comp["n_nonzero"]
    out = dict(comp)
    if m == 0:
        out.update({"p_value": 1.0, "method": "exact", "alternative": alternative})
        return out

    nz = [d for d in _clean(diffs) if d != 0]
    dranks = [int(round(2 * r)) for r in rankdata_average([abs(d) for d in nz])]
    total = sum(dranks)
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in dranks:
        for s in range(total, r - 1, -1):
            counts[s] += counts[s - r]
    n_assign = 2 ** m
    obs = int(round(2 * comp["t_plus"]))
    p_greater = sum(counts[obs:]) / n_assign
    p_less = sum(counts[: obs + 1]) / n_assign
    if alternative == "greater":
        p = p_greater
    elif alternative == "less":
        p = p_less
    else:
        p = min(1.0, 2 * min(p_greater, p_less))
    out.update({"p_value": p, "method": "exact", "alternative": alternative, "n_assignments": n_assign})
    return out


def paired_summary(a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 42) -> dict:
    """
    Paired comparison of two conditions over the same scenarios.
    difference_i = a_i - b_i. Pass a = baseline rate, b = VERIFY-FIN rate
    so that a positive difference means VERIFY-FIN is lower (better).
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and x == x and y == y]
    diffs = [x - y for x, y in pairs]
    summ = mean_sd_ci(diffs)
    rng = random.Random(seed)
    boots = []
    if diffs:
        for _ in range(n_boot):
            samp = [diffs[rng.randrange(len(diffs))] for _ in diffs]
            boots.append(sum(samp) / len(samp))
        boots.sort()
        lo, hi = boots[int(round(0.025 * (len(boots) - 1)))], boots[int(round(0.975 * (len(boots) - 1)))]
    else:
        lo = hi = float("nan")
    return {"diffs": diffs, "n": len(diffs), "summary": summ, "boot_lo": lo, "boot_hi": hi,
            "n_positive": sum(1 for d in diffs if d > 0), "n_negative": sum(1 for d in diffs if d < 0),
            "n_tied": sum(1 for d in diffs if d == 0)}


_Z_95 = 1.959963984540054  # exact standard-normal 97.5th percentile (the rounded 1.96 differs by ~4e-5)


def wilson_interval(successes: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return float("nan")
    p_o = (tp + tn) / n
    p_e = ((tp + fn) / n) * ((tp + fp) / n) + ((fp + tn) / n) * ((fn + tn) / n)
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    m = b + c
    if m == 0:
        return 1.0
    k = max(b, c)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(k, m + 1)) / 2**m)
