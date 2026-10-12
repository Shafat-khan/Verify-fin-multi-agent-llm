"""
Does the v2 text-consistency check misfire on REAL analyst output?

WHY THIS IS NEEDED
    In the fault-injection experiment the source excerpt of an UNcorrupted claim is reconstructed FROM the claim
    text, so the text and the excerpt agree by construction. That makes the experiment's false-flag rate for the
    text check a lower bound. In the live pipeline the excerpt comes from the SEC/FRED data and the sentence comes
    from the LLM's phrasing step, so they could disagree for innocent reasons (rounding style, a different date
    convention). This script measures that directly: it re-runs the Fundamental and Macro analysts (their LLM
    phrasing is served from the disk cache whenever the prompt is unchanged, so this normally costs no tokens),
    then applies the text check to every claim with its REAL excerpt. No knowledge base or verification is needed.

    A "mismatch" printed here is either (a) a real phrasing error that VERIFY-FIN v1 would have let through
    (good news for the paper: a real catch), or (b) a false alarm caused by formatting (a rule to fix before v2
    is adopted). Read each one.

USAGE (project root, venv active):
    python scripts/check_text_on_real_claims.py                # all scenarios in the latest results file
    python scripts/check_text_on_real_claims.py --max-scenarios 3
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Callable, Optional

from verifyfin.agents.text_consistency import check_text_against_excerpt
from verifyfin.agents.verification_agent import VerificationAgent


def evaluate_claims(claims: list, scenario: str) -> tuple[list[dict], dict]:
    """Apply the text check to claims that carry a real source excerpt."""
    rows, counts = [], {"ok": 0, "not_stated": 0, "mismatch": 0, "no_excerpt": 0}
    for c in claims:
        excerpt = c.source_reference.excerpt
        value, _ = VerificationAgent._extract_value_and_unit_from_excerpt(excerpt)
        if not excerpt or value is None:
            counts["no_excerpt"] += 1
            continue
        res = check_text_against_excerpt(c.claim_text, value, VerificationAgent._extract_date_from_excerpt(excerpt))
        bucket = "mismatch" if not res.consistent else ("ok" if "ok" in (res.number, res.date) else "not_stated")
        counts[bucket] += 1
        rows.append({"scenario": scenario, "category": getattr(c.claim_category, "value", c.claim_category),
                     "claim_text": c.claim_text, "excerpt": excerpt, "number": res.number, "date": res.date, "result": bucket})
    return rows, counts


def format_summary(rows: list[dict], counts: dict) -> str:
    n = sum(counts[k] for k in ("ok", "not_stated", "mismatch"))
    L = ["=" * 74, "TEXT-CONSISTENCY CHECK ON REAL ANALYST CLAIMS (real excerpts)", "=" * 74,
         f"claims checked: {n}   consistent: {counts['ok']}   nothing to compare: {counts['not_stated']}   "
         f"MISMATCH: {counts['mismatch']}   (no usable excerpt, skipped: {counts['no_excerpt']})"]
    if counts["mismatch"]:
        L.append(f"mismatch rate: {counts['mismatch'] / n:.1%}\n\nEvery mismatch, for you to read (real error or formatting false alarm?):")
        for r in rows:
            if r["result"] == "mismatch":
                L += [f"  [{r['scenario']}] ({r['category']}) number={r['number']} date={r['date']}",
                      f"      claim  : {r['claim_text']}", f"      excerpt: {r['excerpt']}"]
    else:
        L.append("No mismatches: on real claims the check agrees with every excerpt, so enabling it adds no false alarms here.")
    return "\n".join(L)


def _load_scenarios(results_path: str) -> list[tuple[str, str, str]]:
    seen, out = set(), []
    for r in json.load(open(results_path)):
        key = (r["ticker"], r["time_period"], r.get("period_end"))
        if r.get("period_end") and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _real_analysts():
    from verifyfin.agents.fundamental_agent import FundamentalAgent
    from verifyfin.agents.macro_agent import MacroAgent
    from verifyfin.llm.groq_client import GroqClient

    llm = GroqClient(cache_namespace="verify_fin")   # same namespace as the evaluation => cached phrasing is reused
    fundamental = FundamentalAgent()
    fundamental._llm = llm
    macro = MacroAgent(llm_client=llm)

    def run(ticker, label, period_end):
        return (fundamental.analyze(ticker, label, target_period_end=period_end)
                + macro.analyze(ticker, label, target_date=period_end))
    return run


def main(argv=None, analysts_factory: Callable = _real_analysts) -> None:
    ap = argparse.ArgumentParser(description="Apply the v2 text-consistency check to real analyst claims.")
    ap.add_argument("--results")
    ap.add_argument("--max-scenarios", type=int, default=0)
    ap.add_argument("--out", default="outputs/results/text_check_real_claims.json")
    args = ap.parse_args(argv)
    results = args.results or (sorted(glob.glob("outputs/results/evaluation_results_*.json")) or [None])[-1]
    if not results:
        raise SystemExit("No evaluation_results_*.json found; pass --results.")
    scenarios = _load_scenarios(results)
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]
    run = analysts_factory()
    all_rows, totals = [], {"ok": 0, "not_stated": 0, "mismatch": 0, "no_excerpt": 0}
    for ticker, label, period_end in scenarios:
        try:
            rows, counts = evaluate_claims(run(ticker, label, period_end), label)
        except Exception as e:                      # one failing scenario must not hide the rest
            print(f"  {label}: analysts failed ({e}); skipped")
            continue
        all_rows += rows
        for k, v in counts.items():
            totals[k] += v
        print(f"  {label}: {counts}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"counts": totals, "rows": all_rows}, open(args.out, "w"), indent=2)
    print("\n" + format_summary(all_rows, totals) + f"\n\nSaved: {args.out}")


if __name__ == "__main__":
    main()
