"""
Human labelling for the FREE-TEXT baseline (Single-Agent + RAG), and the end-to-end "error left in the
report" comparison across conditions.

WHY THIS EXISTS
    The automated "unsupported rate" of a free-text baseline is NOT comparable with VERIFY-FIN's: the
    verifier receives every structured claim's source excerpt and can search for its exact number, but a
    sentence extracted from free text has no excerpt, so the verifier falls back to cosine similarity alone,
    which the fault-injection ablation showed flags ~86% of CORRECT claims. The defensible comparison is
    therefore human-labelled, and it measures the quantity that matters to a reader:

        ERROR LEFT IN THE REPORT = share of the claims that actually appear in a condition's report
                                   that a human judges NOT SUPPORTED by the evidence.

    verify_fin             -> the claims routed to the report (GROUNDED, or CHALLENGED and upheld)
    unverified_multiagent  -> every raw claim (nothing is filtered; same agents, so the same claims)
    single_agent_rag       -> the claims extracted from its report (a labelled SAMPLE of them)

TWO STEPS
    1) python scripts/baseline_human_eval.py export  [--n 60]
         writes outputs/results/rag_worksheet.csv (BLIND: no verdicts) and rag_answer_key.json.
         Open rag_worksheet.csv in labelling_tool.html, label every row, save as the same file name.
         For each claim the evidence column shows ALL the numbered passages the model was given
         (rebuilt from the scenario's knowledge base), so you judge it against what the model saw,
         not against one chunk the verifier happened to pick.
    2) python scripts/baseline_human_eval.py score
         prints the error left in each report with Wilson 95% intervals.

LABELS: SUPPORTED / CONTRADICTED / UNSUPPORTED / UNVERIFIABLE (anything other than SUPPORTED counts as
"not supported"). Same rubric as the main evaluation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

from verifyfin.evaluation.stats_utils import wilson_interval

VALID_LABELS = {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "UNVERIFIABLE"}
FIELDS = ["claim_id", "ticker", "time_period", "claim_category", "claim_text", "retrieved_evidence", "human_label", "notes"]


# ------------------------------------------------------------------ pure helpers (unit-tested)

def unique_ids(records: list[dict]) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for r in records:
        r = dict(r)
        base = r["claim_id"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            r["claim_id"] = f"{base}__{seen[base]}"
        out.append(r)
    return out


def sample_across_scenarios(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Up to n records, taken round-robin across scenarios so every scenario is represented, then shuffled."""
    rng = random.Random(seed)
    if len(records) <= n:
        picked = list(records)
    else:
        by_sc: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_sc[r["time_period"]].append(r)
        for lst in by_sc.values():
            rng.shuffle(lst)
        order = sorted(by_sc)
        picked = []
        while len(picked) < n and any(by_sc[s] for s in order):
            for s in order:
                if by_sc[s] and len(picked) < n:
                    picked.append(by_sc[s].pop())
    rng.shuffle(picked)
    return picked


def worksheet_rows(sample: list[dict], contexts: dict[str, str]) -> list[dict]:
    return [{
        "claim_id": r["claim_id"], "ticker": r["ticker"], "time_period": r["time_period"],
        "claim_category": r.get("claim_category", ""), "claim_text": r["claim_text"],
        "retrieved_evidence": contexts.get(r["time_period"]) or r.get("retrieved_evidence") or "",
        "human_label": "", "notes": "",
    } for r in sample]


def answer_key(sample: list[dict]) -> dict:
    return {r["claim_id"]: {"condition": "single_agent_rag", "time_period": r["time_period"], "verdict": r.get("verdict"),
                            "grounding_score": r.get("grounding_score"), "is_usable": r.get("is_usable")} for r in sample}


def error_rate(labels: list[str]) -> dict:
    """Share of claims NOT labelled SUPPORTED, with a Wilson 95% interval."""
    n = len(labels)
    bad = sum(1 for x in labels if x != "SUPPORTED")
    lo, hi = wilson_interval(bad, n)
    return {"n": n, "not_supported": bad, "rate": bad / n if n else float("nan"), "lo": lo, "hi": hi}


def normalise_label(x: str) -> str:
    return (x or "").strip().upper()


def end_to_end_rates(main_rows: list[dict], main_key: dict, rag_rows: list[dict]) -> dict:
    """
    main_rows/main_key : the labelled VERIFY-FIN worksheet and its answer key (is_usable per claim)
    rag_rows           : the labelled Single-Agent + RAG worksheet
    """
    lab = {r["claim_id"]: normalise_label(r["human_label"]) for r in main_rows if normalise_label(r["human_label"])}
    usable = [lab[c] for c in lab if main_key.get(c, {}).get("is_usable")]
    out = {
        "unverified_multiagent": error_rate(list(lab.values())),
        "verify_fin": error_rate(usable),
    }
    rag = [normalise_label(r["human_label"]) for r in rag_rows if normalise_label(r["human_label"])]
    out["single_agent_rag"] = error_rate(rag)
    # also: how much of the verifier's work was needed -- claims removed and how many were truly unsupported
    removed = [lab[c] for c in lab if not main_key.get(c, {}).get("is_usable")]
    out["removed_by_verify_fin"] = {"n": len(removed), "truly_unsupported": sum(1 for x in removed if x != "SUPPORTED")}
    return out


def format_scores(res: dict) -> str:
    L = ["=" * 74, "ERROR LEFT IN THE REPORT (human labels; not supported = anything but SUPPORTED)", "=" * 74, ""]
    L.append(f"  {'condition':<24}{'claims in report':>17}{'not supported':>15}{'rate':>9}   Wilson 95% CI")
    for c in ("unverified_multiagent", "single_agent_rag", "verify_fin"):
        r = res[c]
        L.append(f"  {c:<24}{r['n']:>17}{r['not_supported']:>15}{r['rate']:>9.1%}   [{r['lo']:.1%}, {r['hi']:.1%}]")
    rm = res["removed_by_verify_fin"]
    L += ["", f"  VERIFY-FIN removed {rm['n']} claims, of which {rm['truly_unsupported']} were truly not supported "
              f"(and {rm['n'] - rm['truly_unsupported']} were correct claims removed unnecessarily).",
          "", "READING THIS TABLE",
          "  * single_agent_rag is a labelled SAMPLE of its claims; the others cover every labelled claim.",
          "  * unverified_multiagent is assumed to contain the same claims as VERIFY-FIN's raw claims (same agents).",
          "  * Intervals that overlap mean the data cannot separate the conditions; say so rather than ranking them.",
          "  * One annotator labelled all claims; see the limitations section."]
    return "\n".join(L)


# ------------------------------------------------------------------ I/O

def _latest(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No file matches {pattern}")
    return files[-1]


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _scenario_map(results_path: str) -> dict:
    return {r["time_period"]: (r["ticker"], r["period_end"]) for r in json.load(open(results_path)) if r.get("period_end")}


def _rebuild_contexts(scen: dict, time_periods) -> dict:
    """The numbered passages the RAG baseline was given, rebuilt from each scenario's existing knowledge base."""
    from verifyfin.baselines.single_agent_rag_baseline import SingleAgentRagBaseline
    from verifyfin.data_pipeline.vector_store import VectorStore

    helper = SingleAgentRagBaseline(llm_client=object())
    out = {}
    for tp in sorted(set(time_periods)):
        ticker, period_end = scen[tp]
        store = VectorStore(collection_name=f"eval_{ticker}_{period_end}")
        out[tp] = helper._format_context(helper.retrieve_context(store, ticker)) if store.count() else ""
    return out


def export(args, context_builder: Optional[Callable] = None) -> None:
    rag_path = args.claims or _latest("outputs/results/rag_claims_*.jsonl")
    records = unique_ids(_load_jsonl(rag_path))
    if not records:
        raise SystemExit("The RAG claims file is empty. Run the Single-Agent + RAG baseline first.")
    sample = sample_across_scenarios(records, args.n, args.seed)
    results_path = args.results or _latest("outputs/results/evaluation_results_*.json")
    builder = context_builder or _rebuild_contexts
    contexts = builder(_scenario_map(results_path), [r["time_period"] for r in sample])
    out_dir = Path(args.out_dir)
    ws, key = out_dir / "rag_worksheet.csv", out_dir / "rag_answer_key.json"
    with open(ws, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(worksheet_rows(sample, contexts))
    json.dump(answer_key(sample), open(key, "w"), indent=2)
    print(f"{len(records)} claims in {Path(rag_path).name}; sampled {len(sample)} across "
          f"{len({r['time_period'] for r in sample})} scenarios.")
    print(f"BLIND worksheet : {ws}\n  -> label every row in labelling_tool.html, save/overwrite this same file.")
    print(f"Hidden key      : {key}  (do not open until labelling is finished)")


def score(args) -> None:
    out_dir = Path(args.out_dir)
    main_rows = list(csv.DictReader(open(args.main_worksheet, newline="")))
    main_key = json.load(open(args.main_key))
    rag_rows = list(csv.DictReader(open(out_dir / "rag_worksheet.csv", newline="")))
    unlabelled = [r["claim_id"] for r in rag_rows if normalise_label(r["human_label"]) not in VALID_LABELS]
    if unlabelled:
        raise SystemExit(f"{len(unlabelled)} rows in rag_worksheet.csv are unlabelled or have an invalid label "
                         f"(first: {unlabelled[0]!r}). Finish labelling first.")
    print(format_scores(end_to_end_rates(main_rows, main_key, rag_rows)))


def main(argv=None, context_builder: Optional[Callable] = None) -> None:
    ap = argparse.ArgumentParser(description="Human labelling of the free-text baseline + end-to-end error comparison.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--claims"); e.add_argument("--results"); e.add_argument("--n", type=int, default=60)
    e.add_argument("--seed", type=int, default=42); e.add_argument("--out-dir", default="outputs/results")
    s = sub.add_parser("score")
    s.add_argument("--out-dir", default="outputs/results")
    s.add_argument("--main-worksheet", default="outputs/results/human_eval_worksheet.csv")
    s.add_argument("--main-key", default="outputs/results/human_eval_answer_key.json")
    args = ap.parse_args(argv)
    export(args, context_builder) if args.cmd == "export" else score(args)


if __name__ == "__main__":
    main()
