"""
Scoring script for the human spot-check evaluation (Proposal §2.5).

Reads the completed labelling worksheet (human_label column filled in)
and the hidden answer key produced by sample_for_human_eval.py, then
computes the comparison Proposal §2.5 calls for:

    Human ground truth  vs  VERIFY-FIN's own verdict

Two levels of analysis are produced:

1. PRIMARY (binary "is this claim actually correct/usable" framing --
   this is what Accuracy/Precision/Recall/F1/Cohen's Kappa are computed
   on, since these are standard binary-classification metrics and a
   claim's real-world correctness is fundamentally a yes/no question):
       - Human "usable" = SUPPORTED
       - Human "not usable" = CONTRADICTED, UNSUPPORTED, UNVERIFIABLE
       - System "usable" = claim.is_usable (GROUNDED, or CHALLENGED
         with debate_resolution == "upheld")
   Confusion matrix:
       TP = human says usable, system says usable   (system correctly trusted it)
       TN = human says not usable, system says not usable (system correctly rejected it)
       FP = human says NOT usable, system says usable (a hallucination that slipped through)
       FN = human says usable, system says NOT usable (system too conservative / false rejection)

2. DESCRIPTIVE (full 4-class human label vs. system outcome bucket
   cross-tabulation) -- richer qualitative detail for the Discussion
   chapter, not squeezed into binary metrics that don't naturally fit
   a 4-class comparison.

Usage (from the project root, with venv activated):
    python scripts/score_human_eval.py [worksheet.csv] [answer_key.json]

If no paths given, outputs/results/human_eval_worksheet.csv and
outputs/results/human_eval_answer_key.json are used automatically.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from verifyfin.utils.io_utils import resolve_path

_VALID_LABELS = {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "UNVERIFIABLE"}
_HUMAN_USABLE_LABELS = {"SUPPORTED"}


def load_worksheet(path: Path) -> list[dict]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def load_answer_key(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def validate_and_join(worksheet: list[dict], answer_key: dict) -> list[dict]:
    """
    Joins each worksheet row to its answer-key entry by claim_id, and
    validates the human_label. Rows with a blank or invalid label are
    reported and excluded, rather than silently miscounted as any
    particular class.
    """
    joined = []
    unlabelled = 0
    invalid = []

    for row in worksheet:
        claim_id = row["claim_id"]
        label = (row.get("human_label") or "").strip().upper()

        if not label:
            unlabelled += 1
            continue
        if label not in _VALID_LABELS:
            invalid.append((claim_id, label))
            continue
        if claim_id not in answer_key:
            print(f"WARNING: claim_id '{claim_id}' not found in answer key; skipping.")
            continue

        answer = answer_key[claim_id]
        joined.append(
            {
                "claim_id": claim_id,
                "human_label": label,
                "human_usable": label in _HUMAN_USABLE_LABELS,
                "system_usable": bool(answer["is_usable"]),
                "system_outcome_bucket": answer["outcome_bucket"],
                "claim_category": row.get("claim_category", ""),
                "ticker": row.get("ticker", ""),
                "time_period": row.get("time_period", ""),
                "claim_text": row.get("claim_text", ""),
                "notes": row.get("notes", ""),
                "grounding_score": answer.get("grounding_score"),
            }
        )

    if unlabelled:
        print(f"\nNOTE: {unlabelled} row(s) have no human_label yet and were excluded from scoring.")
    if invalid:
        print(f"\nWARNING: {len(invalid)} row(s) have an unrecognised label and were excluded:")
        for claim_id, label in invalid[:10]:
            print(f"  {claim_id}: '{label}' (expected one of {sorted(_VALID_LABELS)})")

    return joined


def compute_binary_metrics(joined: list[dict]) -> dict:
    tp = sum(1 for r in joined if r["human_usable"] and r["system_usable"])
    tn = sum(1 for r in joined if not r["human_usable"] and not r["system_usable"])
    fp = sum(1 for r in joined if not r["human_usable"] and r["system_usable"])
    fn = sum(1 for r in joined if r["human_usable"] and not r["system_usable"])

    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")

    # Cohen's kappa: kappa = (p_o - p_e) / (1 - p_e)
    # p_o = observed agreement proportion (= accuracy for a 2x2 table)
    # p_e = expected agreement by chance, from each rater's marginal proportions
    p_o = accuracy
    human_usable_rate = (tp + fn) / n if n else float("nan")
    system_usable_rate = (tp + fp) / n if n else float("nan")
    p_e = (human_usable_rate * system_usable_rate) + (
        (1 - human_usable_rate) * (1 - system_usable_rate)
    )
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else float("nan")

    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohens_kappa": kappa,
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score 95% interval for a proportion. Preferred over the normal
    approximation here because several proportions are small and n is
    modest (the normal interval can dip below 0 or collapse to zero width
    at p=0, which is exactly the case for a low residual-error rate).
    """
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compute_in_report_error_rates(joined: list[dict]) -> dict:
    """
    The comparison the research question actually asks for, using the
    INDEPENDENT human labels (not the system's own checker, which would
    make VERIFY-FIN's in-report error 0% by construction):

      all_claims   -- share of ALL claims a human judges not SUPPORTED. This is
                      what an unverified multi-agent report would contain, since
                      it passes every claim through unfiltered.
      kept_claims  -- share not SUPPORTED among only the claims VERIFY-FIN let
                      through to its report (system_usable).

    The two groups overlap (kept is a subset of all), so they are reported
    with separate intervals and an absolute difference, not compared with a
    naive two-independent-proportions test.
    """
    n_all = len(joined)
    bad_all = sum(1 for r in joined if not r["human_usable"])
    kept = [r for r in joined if r["system_usable"]]
    n_kept = len(kept)
    bad_kept = sum(1 for r in kept if not r["human_usable"])
    removed = [r for r in joined if not r["system_usable"]]
    n_removed = len(removed)
    bad_removed = sum(1 for r in removed if not r["human_usable"])

    return {
        "n_all": n_all,
        "bad_all": bad_all,
        "rate_all": bad_all / n_all if n_all else float("nan"),
        "ci_all": wilson_interval(bad_all, n_all),
        "n_kept": n_kept,
        "bad_kept": bad_kept,
        "rate_kept": bad_kept / n_kept if n_kept else float("nan"),
        "ci_kept": wilson_interval(bad_kept, n_kept),
        "n_removed": n_removed,
        "bad_removed": bad_removed,
    }


def print_in_report_comparison(joined: list[dict]) -> None:
    r = compute_in_report_error_rates(joined)
    print("\n" + "=" * 70)
    print("RESEARCH QUESTION: unsupported claims in the report, judged by humans")
    print("=" * 70)
    print(
        "\nValid as a population estimate only if the worksheet covers (nearly) ALL claims, or a "
        "proportionally weighted sample. sample_for_human_eval.py uses the full population when "
        "it has fewer claims than the target size; a stratified sample is NOT proportional."
    )

    def fmt(rate, ci):
        return f"{rate:.1%}  (Wilson 95% CI [{ci[0]:.1%}, {ci[1]:.1%}])"

    print(f"\n  All claims (what an unverified report would contain):   n={r['n_all']:<4} not-supported = {fmt(r['rate_all'], r['ci_all'])}")
    print(f"  Claims VERIFY-FIN kept (what its report contains):      n={r['n_kept']:<4} not-supported = {fmt(r['rate_kept'], r['ci_kept'])}")
    if r["n_all"] and r["n_kept"]:
        print(f"\n  Absolute reduction in unsupported share of the report: {r['rate_all'] - r['rate_kept']:+.1%}")
    print(
        f"\n  Of the {r['n_removed']} claims VERIFY-FIN removed, humans judged {r['bad_removed']} genuinely not supported "
        f"and {r['n_removed'] - r['bad_removed']} supported (the latter are false rejections: correct claims lost)."
    )


def print_binary_report(metrics: dict) -> None:
    print("\n" + "=" * 70)
    print("PRIMARY ANALYSIS: Human ground truth vs. VERIFY-FIN (binary usable/not-usable)")
    print("=" * 70)

    print(f"\nn = {metrics['n']} labelled claims\n")
    print("Confusion matrix:")
    print(f"                        System: usable   System: not usable")
    print(f"  Human: usable              {metrics['tp']:>4}                {metrics['fn']:>4}")
    print(f"  Human: not usable          {metrics['fp']:>4}                {metrics['tn']:>4}")

    print(f"\nAccuracy:      {metrics['accuracy']:.3f}")
    print(f"Precision:     {metrics['precision']:.3f}   (of claims the system trusted, how many were actually correct)")
    print(f"Recall:        {metrics['recall']:.3f}   (of claims that were actually correct, how many the system trusted)")
    print(f"F1:            {metrics['f1']:.3f}")
    print(f"Cohen's kappa: {metrics['cohens_kappa']:.3f}")
    print(_interpret_kappa(metrics["cohens_kappa"]))

    if metrics["fp"] > 0:
        print(
            f"\n{metrics['fp']} claim(s) were FALSE POSITIVES -- VERIFY-FIN marked them usable but a human "
            f"judged them not supported. These are the genuine hallucinations that slipped through the "
            f"pipeline and are worth discussing individually in your Results/Discussion chapter."
        )


def _interpret_kappa(kappa: float) -> str:
    if kappa != kappa:  # NaN check
        return ""
    # Landis & Koch (1977) benchmark scale, commonly cited for Cohen's kappa interpretation
    if kappa < 0:
        band = "poor (worse than chance)"
    elif kappa < 0.20:
        band = "slight"
    elif kappa < 0.40:
        band = "fair"
    elif kappa < 0.60:
        band = "moderate"
    elif kappa < 0.80:
        band = "substantial"
    else:
        band = "almost perfect"
    return f"  (Landis & Koch, 1977 benchmark: {band} agreement)"


def _fmt(x: float, pct: bool = False) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.3f}"


def print_metric_intervals(metrics: dict) -> None:
    """Wilson 95% intervals for the headline proportions (n is modest, so report uncertainty, not just point values)."""
    n, tp, tn, fp, fn = metrics["n"], metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]
    print("\n95% Wilson intervals:")
    for name, k, d in [("Accuracy", tp + tn, n), ("Precision", tp, tp + fp), ("Recall", tp, tp + fn)]:
        lo, hi = wilson_interval(k, d)
        print(f"  {name:<10} {k}/{d} = {k / d:.1%}   [{lo:.1%}, {hi:.1%}]" if d else f"  {name:<10} n/a")


def mcnemar_exact_vs_accept_all(joined: list[dict]) -> dict:
    """
    Paired comparison on the SAME claims, human labels as ground truth:
    VERIFY-FIN's keep/remove decision vs. an 'accept everything' policy
    (what the Unverified Multi-Agent report does). Discordant claims:
      b = bad claims VERIFY-FIN removed (it is right, accept-all is wrong)
      c = good claims VERIFY-FIN removed (it is wrong, accept-all is right)
    Claims both get wrong (bad claims VERIFY-FIN kept) or both get right
    do not enter the test. Exact two-sided binomial p-value on b vs c.
    """
    b = sum(1 for r in joined if not r["human_usable"] and not r["system_usable"])
    c = sum(1 for r in joined if r["human_usable"] and not r["system_usable"])
    m = b + c
    if m == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = max(b, c)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k, m + 1)) / 2 ** m)
    return {"b": b, "c": c, "p": p}


def print_filter_tradeoff(joined: list[dict]) -> None:
    r = mcnemar_exact_vs_accept_all(joined)
    bad_total = sum(1 for x in joined if not x["human_usable"])
    good_total = len(joined) - bad_total
    print("\n" + "=" * 70)
    print("WHAT THE FILTER BUYS AND WHAT IT COSTS (VERIFY-FIN vs. accept-everything)")
    print("=" * 70)
    caught = r["b"]
    lo, hi = wilson_interval(caught, bad_total)
    print(f"\n  Unsupported claims removed: {caught}/{bad_total} = {_fmt(caught / bad_total if bad_total else float('nan'), True)}"
          f"   (Wilson 95% CI [{lo:.1%}, {hi:.1%}])")
    print(f"  Supported claims lost:      {r['c']}/{good_total} = {_fmt(r['c'] / good_total if good_total else float('nan'), True)}")
    print(f"  McNemar exact test on the {r['b'] + r['c']} discordant claims ({r['b']} bad removed vs {r['c']} good lost): "
          f"two-sided p = {r['p']:.3f}")
    if r["p"] >= 0.05:
        print("  -> Not significant: the net accuracy gain over accept-everything cannot be distinguished from "
              "chance at this sample size. Report the purity gain and the loss rate side by side instead of "
              "claiming a significant accuracy win.")
    else:
        print("  -> Significant at alpha=0.05.")


def print_debate_ablation(joined: list[dict]) -> None:
    """
    Counterfactual ablation using the human labels: what if there were no
    Debate Agent, so every CHALLENGED claim stayed unresolved and was dropped?
    Only GROUNDED claims would be usable. The claims themselves are unchanged
    by the Debate step, so this is a valid same-claims comparison.
    """
    no_debate = [dict(r, system_usable=(r["system_outcome_bucket"] == "GROUNDED")) for r in joined]
    a = compute_binary_metrics(no_debate)
    b = compute_binary_metrics(joined)
    print("\n" + "=" * 70)
    print("ABLATION: Verification only (no Debate Agent) vs. full VERIFY-FIN")
    print("=" * 70)
    print(f"\n  {'':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Kappa':>8}")
    for name, m in [("Verification only", a), ("Verification + Debate", b)]:
        print(f"  {name:<28}{_fmt(m['accuracy']):>10}{_fmt(m['precision']):>11}{_fmt(m['recall']):>9}{_fmt(m['f1']):>8}{_fmt(m['cohens_kappa']):>8}")
    upheld = [r for r in joined if r["system_outcome_bucket"] == "UPHELD"]
    rejected = [r for r in joined if r["system_outcome_bucket"] == "REJECTED"]
    up_ok = sum(1 for r in upheld if r["human_usable"])
    rej_bad = sum(1 for r in rejected if not r["human_usable"])
    print(f"\n  Debate upheld {len(upheld)} claims; humans judged {up_ok} of them SUPPORTED.")
    print(f"  Debate rejected {len(rejected)} claims; humans judged {rej_bad} genuinely not supported "
          f"and {len(rejected) - rej_bad} supported (false rejections).")


def print_per_category(joined: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("BY CLAIM CATEGORY (human labels vs. VERIFY-FIN)")
    print("=" * 70)
    print(f"\n  {'category':<11}{'n':>5}{'human OK':>10}{'kept':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'acc':>8}")
    for cat in sorted({r["claim_category"] for r in joined}):
        sub = [r for r in joined if r["claim_category"] == cat]
        m = compute_binary_metrics(sub)
        human_ok = sum(1 for r in sub if r["human_usable"])
        kept = sum(1 for r in sub if r["system_usable"])
        print(f"  {cat:<11}{m['n']:>5}{human_ok:>10}{kept:>6}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}{m['tn']:>5}{_fmt(m['accuracy']):>8}")
    print("\n  Small categories (e.g. sentiment, one claim per scenario) have wide uncertainty; do not over-read them.")


def print_error_analysis(joined: list[dict]) -> None:
    def show(title: str, items: list[dict], note: str) -> None:
        print(f"\n{title} ({len(items)})")
        print(f"  {note}")
        for r in items:
            score = r.get("grounding_score")
            score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            text = (r["claim_text"] or "").replace("\n", " ")
            text = text if len(text) <= 150 else text[:147] + "..."
            print(f"  - [{r['ticker']} {r['time_period']} | {r['claim_category']} | human={r['human_label']} | "
                  f"system={r['system_outcome_bucket']} score={score_txt}]")
            print(f"      {text}")
            if (r.get("notes") or "").strip():
                print(f"      note: {r['notes'].strip()}")

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS: the individual claims behind the confusion matrix")
    print("=" * 70)
    fps = [r for r in joined if r["system_usable"] and not r["human_usable"]]
    fns = [r for r in joined if not r["system_usable"] and r["human_usable"]]
    show("FALSE POSITIVES: kept by VERIFY-FIN, judged not supported by you", fps,
         "Genuine errors that reached the report. Check each: wrong period? wrong series? a near-miss number?")
    show("FALSE REJECTIONS: removed by VERIFY-FIN, judged supported by you", fns,
         "Good claims lost. Check each: was the right evidence retrieved? is it a sentiment judgement call?")


def _percentile_interval(values: list[float], lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        return float("nan"), float("nan")
    return vals[int(round(lo / 100 * (len(vals) - 1)))], vals[int(round(hi / 100 * (len(vals) - 1)))]


def print_cluster_bootstrap(joined: list[dict], n_boot: int = 5000, seed: int = 42) -> None:
    """
    Cluster-aware uncertainty. The 171 claims are NOT independent: sentiment
    claims are identical across a ticker's scenarios (current headlines), and
    macro claims are identical across tickers sharing a date. Wilson intervals
    treat every row as independent and are therefore too narrow.

    Rows with the same (claim_category, claim_text) form one cluster. The
    bootstrap resamples whole clusters with replacement, so a claim that
    repeats counts as one draw, not several. Percentile 95% intervals.
    """
    clusters: dict[tuple, list[dict]] = defaultdict(list)
    for r in joined:
        clusters[(r["claim_category"], (r["claim_text"] or "").strip())].append(r)
    groups = list(clusters.values())

    print("\n" + "=" * 70)
    print("CLUSTER-AWARE UNCERTAINTY (identical claims resampled together)")
    print("=" * 70)
    per_cat: dict[str, set] = defaultdict(set)
    rows_per_cat: dict[str, int] = defaultdict(int)
    for (cat, text) in clusters:
        per_cat[cat].add(text)
    for r in joined:
        rows_per_cat[r["claim_category"]] += 1
    print(f"\n  Distinct claims vs. rows: " + ", ".join(
        f"{cat} {len(per_cat[cat])}/{rows_per_cat[cat]}" for cat in sorted(per_cat)))
    print(f"  Total distinct claims (clusters): {len(groups)} of {len(joined)} rows")

    rng = random.Random(seed)
    acc, kap, r_all, r_kept, diff = [], [], [], [], []
    for _ in range(n_boot):
        sample: list[dict] = []
        for _ in range(len(groups)):
            sample.extend(groups[rng.randrange(len(groups))])
        m = compute_binary_metrics(sample)
        ir = compute_in_report_error_rates(sample)
        acc.append(m["accuracy"]); kap.append(m["cohens_kappa"])
        r_all.append(ir["rate_all"]); r_kept.append(ir["rate_kept"])
        if ir["rate_all"] == ir["rate_all"] and ir["rate_kept"] == ir["rate_kept"]:
            diff.append(ir["rate_all"] - ir["rate_kept"])

    point = compute_binary_metrics(joined)
    pir = compute_in_report_error_rates(joined)
    print(f"\n  {n_boot} bootstrap resamples, seed {seed}. Point estimate [95% cluster-bootstrap interval]:")
    for name, pt, vals, pct in [
        ("Accuracy", point["accuracy"], acc, True),
        ("Cohen's kappa", point["cohens_kappa"], kap, False),
        ("Unsupported share, all claims", pir["rate_all"], r_all, True),
        ("Unsupported share, kept claims", pir["rate_kept"], r_kept, True),
        ("Reduction (all - kept)", pir["rate_all"] - pir["rate_kept"], diff, True),
    ]:
        lo, hi = _percentile_interval(vals)
        f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
        print(f"    {name:<32}{f(pt):>8}   [{f(lo)}, {f(hi)}]")
    print("\n  Report THESE intervals (not the Wilson ones) as the uncertainty on the headline figures.")


def print_descriptive_crosstab(joined: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("DESCRIPTIVE: Human label (4-class) vs. VERIFY-FIN outcome bucket")
    print("=" * 70)

    crosstab: dict[str, Counter] = defaultdict(Counter)
    for r in joined:
        crosstab[r["human_label"]][r["system_outcome_bucket"]] += 1

    buckets = sorted({r["system_outcome_bucket"] for r in joined})
    print(f"\n{'':<15}" + "".join(f"{b:>12}" for b in buckets))
    for label in sorted(_VALID_LABELS):
        row = crosstab.get(label, Counter())
        print(f"{label:<15}" + "".join(f"{row.get(b, 0):>12}" for b in buckets))


def main() -> None:
    worksheet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else resolve_path("outputs/results/human_eval_worksheet.csv")
    answer_key_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else resolve_path("outputs/results/human_eval_answer_key.json")
    )

    print(f"Loading worksheet from: {worksheet_path}")
    print(f"Loading answer key from: {answer_key_path}")

    worksheet = load_worksheet(worksheet_path)
    answer_key = load_answer_key(answer_key_path)

    joined = validate_and_join(worksheet, answer_key)

    if not joined:
        print("\nNo validly labelled claims found. Fill in the 'human_label' column and re-run.")
        return

    metrics = compute_binary_metrics(joined)
    print_binary_report(metrics)
    print_metric_intervals(metrics)
    print_in_report_comparison(joined)
    print_filter_tradeoff(joined)
    print_debate_ablation(joined)
    print_per_category(joined)
    print_cluster_bootstrap(joined)
    print_error_analysis(joined)
    print_descriptive_crosstab(joined)


if __name__ == "__main__":
    main()
