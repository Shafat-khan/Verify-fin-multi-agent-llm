"""
Statistical analysis of evaluation results (Proposal §2.5, "Result
Analysis" activities: quantitative comparison tables and statistical
analysis).

Three-way hallucination-rate comparison: VERIFY-FIN's own
hallucination_rate() reflects real, pipeline-gating verification. The
two baseline conditions don't produce that number the same way by
design (Unverified Multi-Agent never verifies anything; Single-Agent
produces free text with no claim structure at all) -- so the
evaluation harness (scripts/run_evaluation.py) now also runs a
uniform, measurement-only post-hoc fact-check on both baselines'
output, using the exact same Verification + Debate pipeline VERIFY-FIN
uses, against the same scenario's knowledge base. Those posthoc_*
fields are what analyze_three_way_hallucination_comparison() below
reports as the fair, apples-to-apples headline number.

Usage (from the project root, with venv activated):
    python scripts/analyze_results.py [path/to/evaluation_results.json]

If no path is given, the most recent evaluation_results_*.json in
outputs/results/ is used automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from verifyfin.evaluation.stats_utils import mean_sd_ci, paired_summary, wilcoxon_signflip

from verifyfin.utils.io_utils import resolve_path


def find_latest_results_file() -> Path:
    results_dir = resolve_path("outputs/results")
    candidates = sorted(results_dir.glob("evaluation_results_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No evaluation_results_*.json files found in {results_dir}. "
            f"Run scripts/run_evaluation.py first."
        )
    return candidates[-1]  # lexicographic sort of the timestamp suffix == chronological order


def load_results(path: Path) -> pd.DataFrame:
    with open(path, "r") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    return df


def mean_and_ci(values: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, ci_lower, ci_upper) using a t-distribution -- appropriate for small samples (n<30)."""
    values = values.dropna()
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(values.iloc[0]), float("nan"), float("nan")
    mean = values.mean()
    sem = stats.sem(values)
    margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def analyze_verify_fin_hallucination_rate(df: pd.DataFrame) -> None:
    print_section("VERIFY-FIN Hallucination Rate (primary evaluation metric)")

    vf = df[df["condition"] == "verify_fin"].copy()
    vf["hallucination_rate"] = 1.0 - (vf["usable_claims_count"] / vf["total_claims"])

    summ = mean_sd_ci(vf["hallucination_rate"])
    print(f"\nAcross {len(vf)} scenarios (scenario-level mean; each scenario weighted equally):")
    print(f"  Mean unsupported rate : {summ['mean']:.2%}")
    print(f"  SD                    : {summ['sd']:.2%}   <- spread of the scenarios themselves")
    print(f"  SE (SD/sqrt(n))       : {summ['se']:.2%}   <- precision of the mean")
    print(f"  95% CI (t, n-1 d.f.)  : [{summ['lo']:.2%}, {summ['hi']:.2%}]   = mean +/- {summ['half_width']:.2%}")
    pooled = 1.0 - vf["usable_claims_count"].sum() / vf["total_claims"].sum()
    print(f"  Pooled claim-level rate: {pooled:.2%}   ({int(vf['total_claims'].sum() - vf['usable_claims_count'].sum())}/{int(vf['total_claims'].sum())} claims; differs from the scenario mean because scenarios have different claim counts)")
    print(f"  Min: {vf['hallucination_rate'].min():.2%}  |  Max: {vf['hallucination_rate'].max():.2%}")

    print("\nPer-ticker breakdown:")
    per_ticker = vf.groupby("ticker")["hallucination_rate"].agg(["mean", "std", "count"])
    print(per_ticker.to_string(float_format=lambda x: f"{x:.2%}" if not pd.isna(x) else "n/a"))


def analyze_claim_verdicts(df: pd.DataFrame) -> None:
    print_section("Claim Verdict Distribution (VERIFY-FIN)")

    vf = df[df["condition"] == "verify_fin"]
    totals = vf[["grounded_count", "challenged_count", "blocked_count"]].sum()
    total_claims = totals.sum()

    print(f"\nAcross all {len(vf)} scenarios ({int(total_claims)} total claims):")
    for verdict, count in totals.items():
        pct = count / total_claims if total_claims else 0
        print(f"  {verdict.replace('_count', '').upper():>12}: {int(count):>4}  ({pct:.1%})")

    debate_upheld = vf["debate_upheld_count"].sum()
    debate_rejected = vf["debate_rejected_count"].sum()
    debate_total = debate_upheld + debate_rejected
    if debate_total:
        print(f"\nDebate Agent outcomes ({int(debate_total)} claims reviewed):")
        print(f"  Upheld:   {int(debate_upheld):>4}  ({debate_upheld / debate_total:.1%})")
        print(f"  Rejected: {int(debate_rejected):>4}  ({debate_rejected / debate_total:.1%})")


def analyze_runtime_and_cost(df: pd.DataFrame) -> None:
    print_section("Runtime and LLM Usage by Condition (descriptive comparison)")

    summary = df.groupby("condition").agg(
        mean_runtime_seconds=("runtime_seconds", "mean"),
        mean_llm_calls=("llm_call_count", "mean"),
        mean_api_calls=("llm_api_call_count", "mean"),
        mean_tokens=("llm_total_tokens", "mean"),
        scenarios=("runtime_seconds", "count"),
    )
    print()
    print(summary.round(2).to_string())

    print(
        "\nNote: VERIFY-FIN's higher call/token count reflects the additional "
        "Verification + Debate stages -- this is the direct cost of the "
        "hallucination-mitigation mechanism, worth reporting alongside the "
        "hallucination rate itself as a cost/reliability trade-off."
    )


def analyze_three_way_hallucination_comparison(df: pd.DataFrame) -> None:
    """
    The fair, apples-to-apples comparison: VERIFY-FIN's own hallucination
    rate (real, pipeline-gating) against both baselines' POST-HOC
    hallucination rate (the same Verification + Debate pipeline applied
    uniformly, for measurement only, after the fact -- see
    posthoc_factcheck module docstring for why this is necessary).
    """
    print_section("Three-Way Hallucination Rate Comparison (fair, uniform measurement)")

    df = df.copy()
    df["fair_hallucination_rate"] = df.apply(
        lambda row: (1.0 - row["usable_claims_count"] / row["total_claims"])
        if row["condition"] == "verify_fin" and row["total_claims"]
        else (1.0 - row["posthoc_usable_count"] / row["posthoc_total_claims"])
        if pd.notna(row.get("posthoc_total_claims")) and row["posthoc_total_claims"]
        else float("nan"),
        axis=1,
    )

    summary = df.groupby("condition")["fair_hallucination_rate"].agg(["mean", "std", "count"])
    print()
    print(summary.to_string(float_format=lambda x: f"{x:.2%}" if not pd.isna(x) else "n/a"))

    extraction_failures = (df[df["posthoc_extraction_failed"] == True]  # noqa: E712
                           if "posthoc_extraction_failed" in df.columns else df.iloc[0:0])
    if len(extraction_failures) > 0:
        by_cond = extraction_failures.groupby("condition")["time_period"].apply(list) \
            if "time_period" in extraction_failures.columns else extraction_failures.groupby("condition").size()
        for cond, ids in by_cond.items():
            n = len(ids) if isinstance(ids, list) else int(ids)
            listing = f" ({', '.join(map(str, ids))})" if isinstance(ids, list) else ""
            print(f"\nNote: {n} '{cond}' scenario(s){listing} had no claims extractable from their free text "
                  f"and are EXCLUDED from the mean above. Excluding failures is a selection effect: fix the cause "
                  f"(an empty or truncated report) and re-run rather than reporting a mean over the survivors.")

    print(
        "\nHOW TO READ THIS TABLE: each number is the share of a condition's claims that the verifier FLAGS. "
        "For verify_fin and unverified_multiagent the verifier receives every claim's SOURCE EXCERPT, so it can "
        "search for the exact number. The free-text baselines (single_agent, single_agent_rag) carry no excerpt, so "
        "the verifier falls back to semantic similarity alone -- and the fault-injection ablation showed that "
        "similarity-only scoring flags about 86% of CORRECT claims. A high rate for a free-text baseline therefore "
        "mixes real unsupported statements with an instrument that is blind to them being correct. Do not call it "
        "a hallucination rate; the defensible comparison is the human-labelled one (scripts/baseline_human_eval.py: "
        "the error left in each condition's report)."
    )


_COMPARATORS = [
    ("unverified_multiagent", "Unverified Multi-Agent (same agents + same evidence, no verification)"),
    ("single_agent_rag", "Single-Agent + RAG (one LLM call, SAME retrieved evidence)"),
    ("single_agent", "Single-Agent (no evidence)"),
]


def analyze_paired_comparisons(df: pd.DataFrame) -> None:
    """
    Paired comparison of VERIFY-FIN against every baseline present, scenario by scenario.

    SIGN CONVENTION: difference = rate(baseline) - rate(VERIFY-FIN).
    A POSITIVE difference means VERIFY-FIN has the LOWER (better) unsupported rate.

    Statistics (corrected after a metric audit):
      * Wilcoxon signed-rank p-value is the EXACT sign-flip permutation p-value. scipy's
        default silently switches to a normal approximation when there are zeros or ties;
        with few non-zero differences that approximation is badly anti-conservative (it gave
        p = 0.2071 where the exact value is 0.375 for this project's data), so the scipy
        default is printed only as a warning comparison.
      * Effect size: the rank-biserial correlation r = (T+ - T-)/(T+ + T-) (Kerby, 2014)
        from RANK SUMS. The earlier script printed (n+ - n-)/n, which is a sign-count
        quantity, not the rank-biserial; it is now reported separately as "net sign advantage".
    """
    print_section("Paired comparisons: VERIFY-FIN vs each baseline")
    print("\nSign convention: difference = rate(baseline) - rate(VERIFY-FIN); POSITIVE = VERIFY-FIN lower (better).")
    print(
        "\nWhat each comparison ISOLATES (read the baselines in this order):\n"
        "  Single-Agent (no evidence)  vs  Single-Agent + RAG   -> the value of EVIDENCE ACCESS\n"
        "  Single-Agent + RAG          vs  VERIFY-FIN           -> the value of the VERIFICATION PIPELINE (structured claims + verification + debate)\n"
        "  Unverified Multi-Agent      vs  VERIFY-FIN           -> the value of VERIFICATION alone (same agents, same evidence)\n"
        "  The no-evidence Single-Agent is NOT a fair test of verification: it was never given the evidence it is then checked against."
    )

    df = df.copy()
    df["fair_rate"] = df.apply(_fair_hallucination_rate, axis=1)
    vf = df[df["condition"] == "verify_fin"][["ticker", "time_period", "fair_rate"]].rename(columns={"fair_rate": "vf"})

    for cond, label in _COMPARATORS:
        sub = df[df["condition"] == cond][["ticker", "time_period", "fair_rate"]].rename(columns={"fair_rate": "base"})
        if sub.empty:
            continue
        paired = vf.merge(sub, on=["ticker", "time_period"]).dropna(subset=["vf", "base"])
        n = len(paired)
        print(f"\n{'-' * 70}\nVERIFY-FIN vs {label}\n{'-' * 70}")
        print(f"Paired scenarios: {n}")
        if n < 3:
            print("Too few paired scenarios for a meaningful test.")
            continue

        ps = paired_summary(paired["base"].tolist(), paired["vf"].tolist())
        if n <= 20:
            print(f"\n  {'scenario':<16}{'baseline':>10}{'VERIFY-FIN':>12}{'difference':>12}")
            for (_, r), d in zip(paired.iterrows(), ps["diffs"]):
                print(f"  {r['time_period']:<16}{r['base']:>10.2%}{r['vf']:>12.2%}{d:>+12.2%}")
        sm = ps["summary"]
        print(f"\n  VERIFY-FIN lower in {ps['n_positive']}, higher in {ps['n_negative']}, tied in {ps['n_tied']} of {n} scenarios.")
        print(f"  Mean difference : {sm['mean']:+.2%}   (SD of differences {sm['sd']:.2%}, SE {sm['se']:.2%})")
        print(f"  95% CI of mean difference: t-based [{sm['lo']:+.2%}, {sm['hi']:+.2%}];  bootstrap [{ps['boot_lo']:+.2%}, {ps['boot_hi']:+.2%}]")

        w = wilcoxon_signflip(ps["diffs"], "greater")
        print(f"\n  Wilcoxon signed-rank, one-sided (H1: baseline rate > VERIFY-FIN rate), EXACT permutation:")
        print(f"    non-zero pairs = {w['n_nonzero']},  T+ = {w['t_plus']:.1f},  T- = {w['t_minus']:.1f},  p = {w['p_value']:.4f}")
        try:
            scipy_p = stats.wilcoxon(paired["base"], paired["vf"], alternative="greater").pvalue
            note = "  <-- normal approximation, anti-conservative here; do NOT report" if abs(scipy_p - w["p_value"]) > 0.01 else ""
            print(f"    (scipy default: p = {scipy_p:.4f}{note})")
        except ValueError:
            pass
        print(f"    Rank-biserial r = {w['rank_biserial']:.3f}   (from rank sums)")
        print(f"    Net sign advantage = {w['sign_net_advantage']:.3f}   ((n+ - n-)/non-tied pairs)")
        if w["n_nonzero"] and w["p_value"] >= 0.05:
            min_p = 1 / 2 ** w["n_nonzero"]
            note = (f" With {w['n_nonzero']} non-zero pairs the smallest attainable exact one-sided p is {min_p:.4f}, "
                    f"so significance {'is impossible' if min_p >= 0.05 else 'was reachable only with a very consistent effect'}.")
            print(f"    NOT significant at alpha=0.05.{note} This reflects low power / near-identical outcomes, not proof of equivalence.")
        elif w["n_nonzero"]:
            print("    Significant at alpha=0.05.")


def _fair_hallucination_rate(row) -> float:
    if row["condition"] == "verify_fin" and row.get("total_claims"):
        return 1.0 - (row["usable_claims_count"] / row["total_claims"])
    if pd.notna(row.get("posthoc_total_claims")) and row["posthoc_total_claims"]:
        return 1.0 - (row["posthoc_usable_count"] / row["posthoc_total_claims"])
    return float("nan")


def _grounded_rate(row) -> float:
    """
    Share of claims judged GROUNDED. For verify_fin this is its own gating
    verdict. The baselines have no gating verdicts (nothing was verified),
    so their figure comes from the uniform post-hoc check instead --
    reading grounded_count for them would print a meaningless 0.0%.
    """
    if row["condition"] == "verify_fin":
        return row["grounded_count"] / row["total_claims"] if row.get("total_claims") else float("nan")
    posthoc_total = row.get("posthoc_total_claims")
    if pd.notna(posthoc_total) and posthoc_total:
        return row["posthoc_grounded_count"] / posthoc_total
    return float("nan")


def print_final_summary_table(df: pd.DataFrame) -> None:
    """
    Dissertation-ready summary: one row per condition.

    Runtime is reported as the MEDIAN, not the mean: the Groq client's
    retry-with-backoff sleeps inside the timed region, so a few
    rate-limit waits inflate the mean badly (VERIFY-FIN read 123s vs ~13s
    in a run that hit no waits). Tokens and API-call counts are
    unaffected by this and are the cleaner cost measures.
    """
    print_section("FINAL SUMMARY TABLE (dissertation-ready)")

    df = df.copy()
    df["fair_hallucination_rate"] = df.apply(_fair_hallucination_rate, axis=1)
    df["grounded_rate"] = df.apply(_grounded_rate, axis=1)

    summary = df.groupby("condition").agg(
        hallucination_rate_mean=("fair_hallucination_rate", "mean"),
        hallucination_rate_std=("fair_hallucination_rate", "std"),
        grounded_rate_mean=("grounded_rate", "mean"),
        runtime_seconds_median=("runtime_seconds", "median"),
        tokens_mean=("llm_total_tokens", "mean"),
        api_calls_mean=("llm_api_call_count", "mean"),
        n_scenarios=("fair_hallucination_rate", "count"),
    )

    print("\n  (unsupported rate shown as scenario mean, SD across scenarios, and 95% t-CI of the mean)")
    for condition, row in summary.iterrows():
        vals = df.loc[df["condition"] == condition, "fair_hallucination_rate"]
        ms = mean_sd_ci(vals)
        if ms["n"] and ms["mean"] == ms["mean"]:
            ci_txt = f"CI {ms['lo']:.2%}-{ms['hi']:.2%}" if ms["lo"] == ms["lo"] else "CI n/a"
            halluc = f"{ms['mean']:.2%} (SD {ms['sd']:.2%}; {ci_txt})" if ms["sd"] == ms["sd"] else f"{ms['mean']:.2%}"
        else:
            halluc = "n/a"
        grounded = f"{row['grounded_rate_mean']:.1%}" if pd.notna(row["grounded_rate_mean"]) else "n/a"
        print(
            f"  {condition:<22} unsupported={halluc:<42} grounded={grounded:<7} "
            f"tokens={row['tokens_mean']:.0f}  api_calls={row['api_calls_mean']:.1f}  "
            f"median_runtime={row['runtime_seconds_median']:.1f}s  (n={int(row['n_scenarios'])})"
        )
    print(
        "\nNote: 'single_agent' had NO access to the evidence it is checked against, so its rate mostly measures missing "
        "evidence, not missing verification; compare with 'single_agent_rag' (same evidence, no verification) when present."
    )
    print(
        "\nNote: for verify_fin the rate above is the share of raw claims it FILTERED OUT, not the error "
        "remaining in its report; for the baselines it is the share a uniform post-hoc check would flag. "
        "The error left in each condition's report needs independent human labels (score_human_eval.py)."
    )


def analyze_by_category(df: pd.DataFrame) -> None:
    """Per-category outcome breakdown for VERIFY-FIN, from the per-claim records run_evaluation.py persists."""
    print_section("Per-Category Breakdown (VERIFY-FIN, per-claim records)")

    results_dir = resolve_path("outputs/results")
    candidates = sorted(results_dir.glob("claims_*.jsonl"))
    if not candidates:
        print("\nNo claims_*.jsonl found -- re-run scripts/run_evaluation.py to generate per-claim records.")
        return

    claims = pd.read_json(candidates[-1], lines=True)
    print(f"\nLoaded {len(claims)} claims from {candidates[-1].name}\n")

    def outcome(row) -> str:
        verdict = str(row["verdict"]).lower()
        if verdict == "grounded":
            return "GROUNDED"
        if verdict == "blocked":
            return "BLOCKED"
        if verdict == "challenged":
            if row["debate_resolution"] == "upheld":
                return "UPHELD"
            if row["debate_resolution"] == "rejected":
                return "REJECTED"
            return "UNRESOLVED"
        return "UNKNOWN"

    claims["outcome"] = claims.apply(outcome, axis=1)
    table = pd.crosstab(claims["claim_category"], claims["outcome"])
    table["total"] = table.sum(axis=1)
    usable = claims.groupby("claim_category")["is_usable"].mean()
    table["usable_rate"] = (usable * 100).round(1).astype(str) + "%"
    print(table.to_string())


def main() -> None:
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        results_path = find_latest_results_file()

    print(f"Analyzing results from: {results_path}")
    df = load_results(results_path)
    print(f"Loaded {len(df)} result records across {df['condition'].nunique()} conditions "
          f"and {df['ticker'].nunique()} tickers.")

    analyze_verify_fin_hallucination_rate(df)
    analyze_claim_verdicts(df)
    analyze_runtime_and_cost(df)
    analyze_three_way_hallucination_comparison(df)
    analyze_paired_comparisons(df)
    analyze_by_category(df)
    print_final_summary_table(df)

    print_section("Analysis complete")


if __name__ == "__main__":
    main()
