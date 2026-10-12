"""
Does the v3 evidence-alignment verifier help or hurt on REAL analyst claims?

WHY
    Fault injection shows what v3 catches in synthetic corruptions. This script is the other half: it takes the
    claims the live analysts actually produce (real source excerpts, real knowledge bases) and verifies each one
    under v1 (production), v2 (text check) and v3 (evidence alignment), then lists every claim whose outcome changed.
        * a claim that gets WORSE under v3 is a possible false alarm (a rule to fix before v3 is adopted);
        * a claim that gets BETTER is a repaired citation (e.g. the wrong-month FRED chunk).
    It makes NO Debate calls. It re-runs the Fundamental and Macro analysts (fundamental phrasing is served from the
    disk cache; macro claims may need ~1 short LLM call per scenario if the observation date changed with the
    no-look-ahead fix).

USAGE (project root, venv active)
    python scripts/check_v3_on_real_claims.py
    python scripts/check_v3_on_real_claims.py --max-scenarios 3
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Callable

VARIANTS = {
    "v1": {},
    "v2": {"check_text_consistency": True},
    "v3": {"check_text_consistency": True, "date_aligned_search": True, "label_check": True, "filing_period_check": True},
    "v3x": {"check_text_consistency": True, "date_aligned_search": True, "label_check": True, "filing_period_check": True,
            "exact_record_grounds": True},
}
RANK = {"blocked": 0, "challenged": 1, "grounded": 2}


def _verdict(claim) -> str:
    v = claim.verdict
    return v.value if hasattr(v, "value") else str(v)


def verify_all(claims: list, store, scenario: str, make_agent: Callable) -> list[dict]:
    """Verify every claim under every variant (fresh copies, so variants cannot affect each other)."""
    rows = []
    for c in claims:
        row = {"scenario": scenario, "category": getattr(c.claim_category, "value", c.claim_category),
               "claim_text": c.claim_text, "excerpt": c.source_reference.excerpt}
        for name, flags in VARIANTS.items():
            copy = c.model_copy(deep=True)
            make_agent(store, flags).verify_claim(copy)
            row[name] = {"verdict": _verdict(copy), "score": round(copy.grounding_score, 3), "evidence": copy.retrieved_evidence}
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> str:
    L = ["=" * 76, "v3 EVIDENCE-ALIGNMENT CHECK ON REAL ANALYST CLAIMS", "=" * 76, f"claims verified: {len(rows)}", ""]
    L.append(f"  {'variant':<8}{'GROUNDED':>10}{'CHALLENGED':>12}{'BLOCKED':>9}")
    for name in VARIANTS:
        cnt = {k: sum(1 for r in rows if r[name]["verdict"] == k) for k in RANK}
        L.append(f"  {name:<8}{cnt['grounded']:>10}{cnt['challenged']:>12}{cnt['blocked']:>9}")
    for base, new in (("v1", "v3"), ("v2", "v3"), ("v3", "v3x")):
        worse = [r for r in rows if RANK[r[new]["verdict"]] < RANK[r[base]["verdict"]]]
        better = [r for r in rows if RANK[r[new]["verdict"]] > RANK[r[base]["verdict"]]]
        evidence_changed = [r for r in rows if r[new]["evidence"] != r[base]["evidence"] and r not in worse and r not in better]
        L += ["", f"--- {base} -> {new}: {len(worse)} claims got WORSE (possible false alarms), {len(better)} got BETTER, "
                  f"{len(evidence_changed)} kept the verdict but cite a different passage"]
        for tag, group in (("WORSE", worse), ("BETTER", better), ("DIFFERENT PASSAGE", evidence_changed)):
            for r in group:
                L += [f"  [{tag}] [{r['scenario']}] ({r['category']}) {base}: {r[base]['verdict']} {r[base]['score']}  ->  "
                      f"{new}: {r[new]['verdict']} {r[new]['score']}",
                      f"      claim   : {r['claim_text'][:150]}", f"      excerpt : {r['excerpt']}",
                      f"      {base} cites: {(r[base]['evidence'] or '')[:110]!r}", f"      {new} cites: {(r[new]['evidence'] or '')[:110]!r}"]
    L += ["", "HOW TO READ THIS: v3 is safe to adopt only if no claim in a WORSE group is a correct claim. Read each one."]
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


def _real_store(ticker: str, period_end: str):
    from verifyfin.data_pipeline.vector_store import VectorStore
    from verifyfin.evaluation.store_wrappers import CachingStore

    return CachingStore(VectorStore(collection_name=f"eval_{ticker}_{period_end}"))


def _real_agent(store, flags):
    from verifyfin.agents.verification_agent import VerificationAgent

    return VerificationAgent(vector_store=store, **flags)


def main(argv=None, analysts_factory: Callable = _real_analysts, store_factory: Callable = _real_store,
         agent_factory: Callable = _real_agent) -> None:
    ap = argparse.ArgumentParser(description="Verify real analyst claims under v1 / v2 / v3 and list what changed.")
    ap.add_argument("--results")
    ap.add_argument("--max-scenarios", type=int, default=0)
    ap.add_argument("--out", default="outputs/results/v3_real_claims.json")
    args = ap.parse_args(argv)
    results = args.results or (sorted(glob.glob("outputs/results/evaluation_results_*.json")) or [None])[-1]
    if not results:
        raise SystemExit("No evaluation_results_*.json found; pass --results.")
    scenarios = _load_scenarios(results)
    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]
    run = analysts_factory()
    rows: list[dict] = []
    for ticker, label, period_end in scenarios:
        try:
            claims = run(ticker, label, period_end)
            rows += verify_all(claims, store_factory(ticker, period_end), label, agent_factory)
            print(f"  {label}: {len(claims)} claims verified")
        except Exception as e:                      # one failing scenario must not hide the rest
            print(f"  {label}: failed ({e}); skipped")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=2, default=str)
    print("\n" + summarise(rows) + f"\n\nSaved: {args.out}")


if __name__ == "__main__":
    main()
