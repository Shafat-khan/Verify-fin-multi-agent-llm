"""
Fault-injection experiment + ablation study (faculty feedback points 1 and 6).

WHAT IT DOES
  1. Takes the numerical and macro claims from your human-labelled worksheet.
  2. Corrupts the ones you judged SUPPORTED in controlled ways (value changes, typos,
     wrong line item, wrong date/period, wrong series), each provably FALSE
     (checked against XBRL facts / the FRED chunks in the knowledge base).
  3. Runs every verifier variant (full, plain-cosine, no date preference, no
     targeted search, no boost, strict date, and knowledge-base subsets) over the
     uncorrupted originals and the corrupted claims, against the SAME scenario
     knowledge bases the evaluation used (no re-fetching, no LLM calls).
  4. Reports detection rate (corruptions caught) and false-flag rate
     (correct claims wrongly flagged) per fault type, with cluster-aware
     confidence intervals, score separation (AUROC), a threshold sweep, and
     the ablation table.
  5. Optionally (--debate-budget N) sends a random sample of the CHALLENGED
     corrupted claims to the Debate Agent to estimate END-TO-END detection.

COST: verification needs NO API calls. Only --debate-budget uses the LLM
(roughly 450 tokens per debated claim; the daily Groq limit is 200,000 tokens).

PREREQUISITES: data/chroma_store still contains the eval_<ticker>_<period_end>
collections from your final run (do not delete it). If a collection is
empty, add --rebuild-kb to rebuild it from SEC/FRED/news (network needed).

Usage (project root, venv active):
    python scripts/run_fault_injection.py --dry-run
    python scripts/run_fault_injection.py --quick 30
    python scripts/run_fault_injection.py
    python scripts/run_fault_injection.py --debate-budget 60
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from verifyfin.evaluation import fault_injection as fi
from verifyfin.evaluation import fault_runner as fr
from verifyfin.evaluation.fault_validators import make_validator


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Fault-injection + ablation experiment for the Verification Agent.")
    ap.add_argument("--results"); ap.add_argument("--worksheet", default="outputs/results/human_eval_worksheet.csv")
    ap.add_argument("--answer-key", default="outputs/results/human_eval_answer_key.json")
    ap.add_argument("--out-dir", default="outputs/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", type=int, default=0, help="corrupt only N randomly chosen supported claims (fast smoke test)")
    ap.add_argument("--configs", nargs="*", default=[fr.cfg_name(v, k) for v, k in fr.DEFAULT_CONFIGS],
                    help="verifier|knowledge-base pairs, e.g. full|all strict_date|all full|filing_only")
    ap.add_argument("--v2", action="store_true",
                    help="also run the v2 text-consistency configurations (text_check, text_check_strict, text_check_block)")
    ap.add_argument("--v3", action="store_true",
                    help="also run the v3 evidence-alignment configurations (v3_date, v3_label, v3_period, v3_aligned, v3_exact)")
    ap.add_argument("--debate-config", default="full|all",
                    help="which configuration's CHALLENGED claims go to the Debate Agent (e.g. text_check|all); "
                         "for a non-production config the CHALLENGED originals are debated too")
    ap.add_argument("--operators", nargs="*", help="restrict to these corruption operators")
    ap.add_argument("--no-validate", action="store_true", help="skip XBRL/FRED validation (corruptions may occasionally be TRUE claims)")
    ap.add_argument("--debate-budget", type=int, default=0, help="number of CHALLENGED corrupted claims to send to the Debate Agent")
    ap.add_argument("--rebuild-kb", action="store_true", help="rebuild an empty scenario knowledge base (needs network)")
    ap.add_argument("--n-boot", type=int, default=1500)
    ap.add_argument("--dry-run", action="store_true", help="print the experiment size and exit; touches no knowledge base")
    return ap.parse_args(argv)


def _latest_results() -> str:
    files = sorted(glob.glob("outputs/results/evaluation_results_*.json"))
    if not files:
        raise SystemExit("No evaluation_results_*.json found; pass --results.")
    return files[-1]


def load_scenarios(results_path: str) -> dict:
    scen = {}
    for r in json.load(open(results_path)):
        if r.get("period_end"):
            scen[r["time_period"]] = (r["ticker"], r["period_end"])
    return scen


class RealDeps:
    """Everything that touches the outside world: the knowledge bases, XBRL truth, and the Debate LLM."""

    def __init__(self, scen: dict, rebuild: bool = False):
        self.scen, self.rebuild = scen, rebuild
        self._stores: dict = {}
        self._xbrl = None
        self._truth: dict = {}
        self._scenarios_for_rebuild = None

    def get_store(self, tp: str):
        if tp in self._stores:
            return self._stores[tp]
        from verifyfin.data_pipeline.vector_store import VectorStore

        ticker, period_end = self.scen[tp]
        store = VectorStore(collection_name=f"eval_{ticker}_{period_end}")
        if store.count() == 0:
            if not self.rebuild:
                raise SystemExit(f"Knowledge base 'eval_{ticker}_{period_end}' is empty or missing. Do not delete data/chroma_store "
                                 f"between the evaluation and this experiment; or re-run with --rebuild-kb (needs network).")
            store = self._rebuild(tp)
        self._stores[tp] = store
        return store

    def _rebuild(self, tp: str):
        spec = importlib.util.spec_from_file_location("run_evaluation", Path(__file__).parent / "run_evaluation.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if self._scenarios_for_rebuild is None:
            self._scenarios_for_rebuild = mod.resolve_scenarios(sorted({t for t, _ in self.scen.values()}))
        sc = next((s for s in self._scenarios_for_rebuild if s["time_period_label"] == tp), None)
        if sc is None:
            raise SystemExit(f"Could not resolve scenario {tp} to rebuild its knowledge base.")
        return mod.build_knowledge_base_for_scenario(sc)

    def xbrl_truth(self, ticker: str, period_end: str) -> Optional[dict]:
        key = (ticker, period_end)
        if key in self._truth:
            return self._truth[key]
        try:
            if self._xbrl is None:
                from verifyfin.data_pipeline.edgar_xbrl_client import EdgarXbrlClient
                self._xbrl = EdgarXbrlClient()
            facts = self._xbrl.get_fundamentals_for_period(ticker, period_end)
            self._truth[key] = {k: v for k, v in facts.items() if isinstance(v, dict)}
        except Exception:
            self._truth[key] = None
        return self._truth[key]

    def validator_factory(self, tp: str):
        ticker, period_end = self.scen[tp]
        known = [pe for t, pe in self.scen.values() if t == ticker]
        return make_validator(ticker=ticker, period_end=period_end, store=self.get_store(tp), xbrl_truth=self.xbrl_truth, known_period_ends=known)

    def debate_agent(self):
        from verifyfin.agents.debate_agent import DebateAgent
        from verifyfin.llm.groq_client import GroqClient

        return DebateAgent(llm_client=GroqClient(cache_namespace="fault_injection_debate"))


def main(argv=None, deps_factory: Callable[..., object] = RealDeps) -> None:
    args = parse_args(argv)
    results_path = args.results or _latest_results()
    scen = load_scenarios(results_path)
    worksheet = list(csv.DictReader(open(args.worksheet, newline="")))
    answer_key = json.load(open(args.answer_key))
    period_ends = {tp: pe for tp, (_, pe) in scen.items()}

    bases, skipped = fr.build_base_claims(worksheet, answer_key, period_ends)
    n_sup = sum(1 for b in bases if b["human_supported"])
    print(f"Loaded {len(worksheet)} worksheet rows -> {len(bases)} numerical+macro base claims ({n_sup} judged SUPPORTED); skipped {skipped}")
    if args.dry_run:
        est = sum(len(fi.generate_variants(b, seed=args.seed, operators=args.operators)[0]) for b in bases if b["human_supported"])
        extra = set()
        if args.v2:
            extra |= {fr.cfg_name(v, k) for v, k in fr.V2_CONFIGS}
        if args.v3:
            extra |= {fr.cfg_name(v, k) for v, k in fr.V3_CONFIGS}
        n_cfg = len(set(args.configs) | extra)
        print(f"DRY RUN: about {est} corrupted variants before validation; {n_cfg} verifier configurations -> "
              f"~{(est + len(bases)) * n_cfg} verifications (no API calls). Debate budget {args.debate_budget} "
              f"(~{args.debate_budget * 450} tokens).")
        return

    cfg_args = list(args.configs)
    if args.v2:
        cfg_args += [fr.cfg_name(v, k) for v, k in fr.V2_CONFIGS if fr.cfg_name(v, k) not in cfg_args]
    if args.v3:
        cfg_args += [fr.cfg_name(v, k) for v, k in fr.V3_CONFIGS if fr.cfg_name(v, k) not in cfg_args]
    if args.debate_config not in cfg_args:
        cfg_args.append(args.debate_config)
    configs = []
    for c in cfg_args:
        v, k = c.split("|")
        if v not in fr.VERIFIER_VARIANTS:
            raise SystemExit(f"unknown verifier variant '{v}'; choose from {sorted(fr.VERIFIER_VARIANTS)}")
        configs.append((v, k))
    if fr.PRIMARY not in [fr.cfg_name(*c) for c in configs]:
        configs.insert(0, ("full", "all"))

    deps = deps_factory(scen, rebuild=args.rebuild_kb) if deps_factory is RealDeps else deps_factory(scen)
    debate = deps.debate_agent() if args.debate_budget > 0 else None
    res = fr.run_experiment(
        bases, deps.get_store, configs=configs, seed=args.seed,
        validator_factory=None if args.no_validate else deps.validator_factory,
        max_bases=args.quick or None, debate_agent=debate, debate_budget=args.debate_budget, operators=args.operators,
        debate_config=args.debate_config)

    text = fr.format_report(res, n_boot=args.n_boot)
    print("\n" + text)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fr.save_rows_csv(res["rows"], out / f"fault_injection_rows_{ts}.csv")
    (out / f"fault_injection_report_{ts}.txt").write_text(text)
    json.dump({"configs": res["configs"], "drops": res["drops"], "fidelity": res["fidelity"], "n_variants": len(res["variants"]),
               "n_bases": len(res["bases"]), "seed": args.seed, "validated": not args.no_validate, "quick": args.quick},
              open(out / f"fault_injection_meta_{ts}.json", "w"), indent=2, default=str)
    print(f"\nSaved: fault_injection_report_{ts}.txt, fault_injection_rows_{ts}.csv, fault_injection_meta_{ts}.json in {out}/")


if __name__ == "__main__":
    main()
