"""
Evaluation harness (Proposal §2.5): runs all THREE evaluation
conditions -- Single-Agent baseline, Unverified Multi-Agent baseline,
and full VERIFY-FIN -- across 5 tickers x 3 historical quarters each
(15 scenarios), and saves every result plus its full RunMetrics record
to disk for statistical analysis.

Reproducibility design: historical quarters are resolved ONCE at the
start, via EdgarXbrlClient.get_available_quarterly_periods(), and
PINNED for the rest of the run. Each scenario gets its OWN isolated
knowledge base (a distinct Chroma collection per ticker+quarter) built
from the SPECIFIC filing matching the pinned accession number.

Fair cost comparison (cache_namespace): each condition gets its own
GroqClient cache namespace, so a condition running later in the same
scenario can't silently free-ride on an earlier condition's cached
LLM calls -- this was a real measurement confound found in the first
run (Unverified Multi-Agent appeared to use MORE tokens than the
strictly more expensive VERIFY-FIN condition, purely due to cache
sharing). See groq_client module docstring for detail.

Fair 3-way hallucination-rate comparison (post-hoc fact-check): after
each baseline condition completes its own (unverified) run, a SEPARATE
measurement-only pass applies VERIFY-FIN's own Verification + Debate
pipeline to that condition's claims (or, for Single-Agent, to claims
extracted from its free text) against the SAME scenario's knowledge
base. This never changes what the condition actually produced -- it
only measures what a uniform check would find. See
posthoc_factcheck module docstring for detail.

NOTE: this makes a LARGE number of real API calls and will take a
substantial amount of time.

Usage (from the project root, with venv activated):
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from verifyfin.baselines.single_agent_baseline import SingleAgentBaseline
from verifyfin.baselines.single_agent_rag_baseline import SingleAgentRagBaseline
from verifyfin.baselines.unverified_multiagent_baseline import UnverifiedMultiAgentBaseline
from verifyfin.data_pipeline.chunker import Chunker
from verifyfin.data_pipeline.edgar_client import EdgarClient
from verifyfin.data_pipeline.edgar_xbrl_client import EdgarXbrlClient
from verifyfin.data_pipeline.kb_indexer import index_macro_data, index_news_headlines
from verifyfin.data_pipeline.vector_store import VectorStore
from verifyfin.evaluation.posthoc_factcheck import (
    factcheck_single_agent_report,
    factcheck_unverified_baseline_claims,
)
from verifyfin.graph.pipeline import VerifyFinPipeline
from verifyfin.llm.groq_client import GroqClient
from verifyfin.utils.io_utils import load_config, resolve_path
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)


def resolve_scenarios(tickers: list[str], periods_per_ticker: int = 3) -> list[dict]:
    xbrl = EdgarXbrlClient()
    scenarios = []
    for ticker in tickers:
        try:
            periods = xbrl.get_available_quarterly_periods(ticker, max_periods=periods_per_ticker)
        except Exception as e:
            logger.warning(f"Could not resolve periods for {ticker}: {e}")
            continue

        if len(periods) < periods_per_ticker:
            logger.warning(
                f"Only found {len(periods)} valid quarterly periods for {ticker} "
                f"(wanted {periods_per_ticker}); proceeding with what's available."
            )

        for period in periods:
            scenarios.append(
                {
                    "ticker": ticker,
                    "period_end": period["period_end"],
                    "accession_number": period["accession_number"],
                    "fiscal_year": period.get("fiscal_year"),
                    "fiscal_period": period.get("fiscal_period") or "",
                    "time_period_label": f"{ticker}-{period['calendar_quarter_label']}",
                }
            )
    return scenarios


def build_knowledge_base_for_scenario(scenario: dict) -> VectorStore:
    ticker = scenario["ticker"]
    collection_name = f"eval_{ticker}_{scenario['period_end']}"
    store = VectorStore(collection_name=collection_name)

    edgar = EdgarClient()
    filings_meta = edgar.get_filings_list(ticker, filing_types=["10-Q", "10-K"], max_filings=12)
    matching = next((f for f in filings_meta if f["accession_number"] == scenario["accession_number"]), None)

    if matching is not None:
        raw_html = edgar.download_filing(matching)
        clean_text = edgar.parse_filing_text(raw_html)
        chunker = Chunker()
        chunks = chunker.chunk_text(
            clean_text, source_document_id=matching["accession_number"], source_type="edgar_10q"
        )
        store.add_chunks(chunks, ticker=ticker)
    else:
        logger.warning(
            f"Could not find filing metadata for {ticker} accession {scenario['accession_number']} "
            f"among the {len(filings_meta)} most recent filings; this scenario's knowledge base "
            f"will lack filing text."
        )

    index_macro_data(store, ticker, as_of=scenario["period_end"])   # no data from after the period end
    index_news_headlines(store, ticker)

    return store


def run_single_agent_condition(scenario: dict, store: VectorStore) -> dict:
    baseline = SingleAgentBaseline(cache_namespace="single_agent")
    report = baseline.run(scenario["ticker"], scenario["time_period_label"])
    metrics = baseline.last_run_metrics

    # Post-hoc fact-check: needs its own GroqClient (separate cache namespace,
    # since claim extraction/debate prompts are a different kind of call
    # entirely from the report-generation prompt above).
    posthoc_llm = GroqClient(cache_namespace="single_agent_posthoc")
    factcheck_single_agent_report(
        report, scenario["ticker"], scenario["time_period_label"], store, posthoc_llm, metrics
    )

    return {"report": report, "metrics": metrics}


def run_single_agent_rag_condition(scenario: dict, store: VectorStore) -> dict:
    """
    Single-Agent + RAG (faculty feedback point 4): ONE LLM call given the SAME retrieved evidence from the
    SAME scenario knowledge base the verifier checks against, but with no structured claims, no verification
    and no debate. Comparing it with the no-evidence Single-Agent isolates the value of EVIDENCE ACCESS;
    comparing it with VERIFY-FIN isolates the value of the verification pipeline.
    """
    baseline = SingleAgentRagBaseline(cache_namespace="single_agent_rag")
    report = baseline.run(scenario["ticker"], scenario["time_period_label"], store, period_end=scenario["period_end"])
    metrics = baseline.last_run_metrics

    posthoc_llm = GroqClient(cache_namespace="single_agent_rag_posthoc")
    claims = factcheck_single_agent_report(
        report, scenario["ticker"], scenario["time_period_label"], store, posthoc_llm, metrics
    )
    return {"report": report, "metrics": metrics, "claims": claims}


def open_existing_knowledge_base(scenario: dict) -> VectorStore:
    """
    Open the scenario's EXISTING knowledge base unchanged: no re-download and no new live headlines, so a
    condition added later is shown exactly the evidence the original conditions saw. Falls back to a
    rebuild only if the collection is empty.
    """
    store = VectorStore(collection_name=f"eval_{scenario['ticker']}_{scenario['period_end']}")
    if store.count() == 0:
        logger.warning(f"Knowledge base for {scenario['time_period_label']} is empty; rebuilding it.")
        return build_knowledge_base_for_scenario(scenario)
    return store


def run_unverified_condition(scenario: dict, store: VectorStore) -> dict:
    baseline = UnverifiedMultiAgentBaseline(cache_namespace="unverified_multiagent")
    state = baseline.run(
        scenario["ticker"],
        scenario["time_period_label"],
        target_period_end=scenario["period_end"],
        target_macro_date=scenario["period_end"],
    )
    metrics = baseline.last_run_metrics

    posthoc_llm = GroqClient(cache_namespace="unverified_multiagent_posthoc")
    factcheck_unverified_baseline_claims(state.raw_claims, store, posthoc_llm, metrics)

    return {"report": state.final_report, "metrics": metrics}


def run_verify_fin_condition(scenario: dict, store: VectorStore) -> dict:
    pipeline = VerifyFinPipeline(vector_store=store, cache_namespace="verify_fin")
    state = pipeline.run(
        scenario["ticker"],
        scenario["time_period_label"],
        target_period_end=scenario["period_end"],
        target_macro_date=scenario["period_end"],
    )
    claims = state.debated_claims if state.debated_claims else state.verified_claims
    return {"report": state.final_report, "metrics": pipeline.last_run_metrics, "claims": claims}


def _claim_to_record(scenario: dict, claim) -> dict:
    """
    Flattens one VERIFY-FIN claim into a JSON-serialisable record for
    the human-evaluation claim pool. Deliberately includes the exact
    evidence the system actually retrieved (retrieved_evidence) --
    without this, a human labeller cannot independently judge
    SUPPORTED/CONTRADICTED/UNSUPPORTED/UNVERIFIABLE the same way the
    system did; they would just be guessing from the claim text alone.
    """
    category = claim.claim_category.value if hasattr(claim.claim_category, "value") else claim.claim_category
    verdict = claim.verdict.value if hasattr(claim.verdict, "value") else claim.verdict
    return {
        "claim_id": f"{scenario['ticker']}_{scenario['time_period_label']}_{claim.claim_text[:40]}",
        "ticker": scenario["ticker"],
        "time_period": scenario["time_period_label"],
        "claim_category": category,
        "claim_text": claim.claim_text,
        "retrieved_evidence": getattr(claim, "retrieved_evidence", None),
        "grounding_score": claim.grounding_score,
        "verdict": verdict,
        "debate_resolution": claim.debate_resolution,
        "is_usable": claim.is_usable_in_final_report(),
    }


def _flatten_result(scenario: dict, condition: str, result: dict) -> dict:
    flat = {
        "ticker": scenario["ticker"],
        "time_period": scenario["time_period_label"],
        "period_end": scenario["period_end"],
        "condition": condition,
        "report": result.get("report"),
    }
    metrics = result.get("metrics")
    if metrics is not None:
        flat.update(metrics.model_dump())
    return flat


def _write_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    fieldnames = sorted({k for r in results for k in r.keys() if k != "report"})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


_RESULTS_TIMESTAMP: Optional[str] = None


def _save_results(results: list[dict], output_dir: Path) -> None:
    global _RESULTS_TIMESTAMP
    if _RESULTS_TIMESTAMP is None:
        _RESULTS_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = output_dir / f"evaluation_results_{_RESULTS_TIMESTAMP}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    csv_path = json_path.with_suffix(".csv")
    _write_csv(results, csv_path)

    print(f"  (saved {len(results)} records so far to {json_path.name} / {csv_path.name})")


def _save_claim_records(claim_records: list[dict], output_dir: Path, prefix: str = "claims") -> None:
    """
    Persists every VERIFY-FIN claim seen so far, one per line (JSONL),
    for the human-evaluation sampling script. Only VERIFY-FIN's claims
    are recorded here -- the baselines' claims are already covered by
    the post-hoc fact-check's own aggregate numbers, and a genuine
    human spot-check is most valuable against the actual system
    VERIFY-FIN gates its own report with.
    """
    global _RESULTS_TIMESTAMP
    if _RESULTS_TIMESTAMP is None:
        _RESULTS_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # prefix "rag_claims" deliberately does NOT match the glob "claims_*.jsonl" that the sampler, analysis and audit
    # scripts use to find VERIFY-FIN's claims, so the baseline's claims can never be mistaken for them.
    path = output_dir / f"{prefix}_{_RESULTS_TIMESTAMP}.jsonl"
    with open(path, "w") as f:
        for record in claim_records:
            f.write(json.dumps(record, default=str) + "\n")


def _load_existing_run(output_dir: Path):
    """
    Loads the most recent results file (and its matching claims file) so a
    --fill-missing run can append to it IN PLACE. Returns
    (results, claim_records, rag_claim_records, timestamp) or None if no results file exists.
    """
    candidates = sorted(output_dir.glob("evaluation_results_*.json"))
    if not candidates:
        return None
    latest = candidates[-1]
    timestamp = latest.stem.replace("evaluation_results_", "")
    with open(latest, "r") as f:
        results = json.load(f)

    claims_path = output_dir / f"claims_{timestamp}.jsonl"
    claim_records: list[dict] = []
    if claims_path.exists():
        with open(claims_path, "r") as f:
            claim_records = [json.loads(line) for line in f if line.strip()]
    rag_path = output_dir / f"rag_claims_{timestamp}.jsonl"
    rag_records: list[dict] = []
    if rag_path.exists():
        with open(rag_path, "r") as f:
            rag_records = [json.loads(line) for line in f if line.strip()]
    return results, claim_records, rag_records, timestamp


def main(fill_missing: bool = False, with_rag_baseline: bool = False, redo_failed: bool = False) -> None:
    """
    Full evaluation by default. With fill_missing=True, loads the latest
    results file and re-runs ONLY the (scenario, condition) pairs that have
    no result row yet -- e.g. a condition that crashed on a rate-limit error
    -- then appends to the same files (same timestamp) instead of starting
    a new run.

    Caveat for filled-in pairs: the knowledge base for that scenario is
    rebuilt, and news headlines are fetched live, so sentiment claims in a
    filled-in pair are checked against headlines from the re-run time, not
    the original run's. Filing text and FRED history are unaffected.

    with_rag_baseline=True adds a fourth condition, "single_agent_rag" (one
    LLM call given the SAME retrieved evidence, no verification). Combined
    with fill_missing it adds ONLY that condition to an existing run, opening
    each scenario's EXISTING knowledge base unchanged (no rebuild), so the
    baseline sees exactly the evidence the other conditions saw. Safe to
    interrupt and re-run: finished scenarios are skipped.

    redo_failed=True (needs fill_missing) first DROPS every baseline result row whose post-hoc claim
    extraction failed (e.g. a Single-Agent report that came back empty), so those scenarios are run again.
    A failed row must be re-run, not averaged around: excluding it from the mean is a selection effect.
    """
    global _RESULTS_TIMESTAMP

    config = load_config()
    tickers = config["tickers"]["evaluation"]

    print(f"Resolving 3 historical quarters for {len(tickers)} tickers ({', '.join(tickers)})...")
    scenarios = resolve_scenarios(tickers, periods_per_ticker=3)
    print(f"Resolved {len(scenarios)} scenarios:\n")
    for s in scenarios:
        print(f"  {s['ticker']}: {s['time_period_label']} (period end {s['period_end']}, accn {s['accession_number']})")

    results: list[dict] = []
    claim_records: list[dict] = []
    rag_claim_records: list[dict] = []
    output_dir = resolve_path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if fill_missing:
        existing = _load_existing_run(output_dir)
        if existing is None:
            raise SystemExit("--fill-missing needs an existing evaluation_results_*.json in outputs/results/.")
        results, claim_records, rag_claim_records, _RESULTS_TIMESTAMP = existing
        print(f"\n--fill-missing: loaded {len(results)} existing result rows and {len(claim_records)} claim records "
              f"(run {_RESULTS_TIMESTAMP}); only missing (scenario, condition) pairs will be run.")

    if redo_failed:
        if not fill_missing:
            raise SystemExit("--redo-failed only makes sense together with --fill-missing.")
        kept = [r for r in results if not (r.get("condition") in ("single_agent", "single_agent_rag")
                                           and str(r.get("posthoc_extraction_failed")).lower() == "true")]
        print(f"--redo-failed: dropping {len(results) - len(kept)} baseline row(s) whose claim extraction failed; "
              f"they will be run again.")
        results = kept

    # Resolved at call time (not module import time) so the runners are looked up by name.
    runners = [
        ("single_agent", run_single_agent_condition),
        ("unverified_multiagent", run_unverified_condition),
        ("verify_fin", run_verify_fin_condition),
    ]
    if with_rag_baseline:
        runners.append(("single_agent_rag", run_single_agent_rag_condition))

    for i, scenario in enumerate(scenarios, 1):
        label = scenario["time_period_label"]
        present = {r["condition"] for r in results if r.get("time_period") == label}
        needed = [name for name, _ in runners if not (fill_missing and name in present)]

        if not needed:
            print(f"\nSCENARIO {i}/{len(scenarios)}: {label} -- all conditions present, skipping.")
            continue

        print(f"\n{'=' * 70}\nSCENARIO {i}/{len(scenarios)}: {scenario['ticker']} {label}  (running: {', '.join(needed)})\n{'=' * 70}")

        reuse_kb = fill_missing and needed == ["single_agent_rag"]
        print("Opening the existing knowledge base for this scenario..." if reuse_kb
              else "Building isolated knowledge base for this scenario...")
        try:
            store = open_existing_knowledge_base(scenario) if reuse_kb else build_knowledge_base_for_scenario(scenario)
        except Exception as e:
            logger.error(f"Failed to build knowledge base for {scenario}: {e}")
            continue

        for name, runner in runners:
            if name not in needed:
                continue
            print(f"Condition: {name} ...")
            try:
                outcome = runner(scenario, store)
                results.append(_flatten_result(scenario, name, outcome))
                if name == "verify_fin":
                    for claim in outcome.get("claims", []):
                        claim_records.append(_claim_to_record(scenario, claim))
                elif name == "single_agent_rag":
                    for claim in outcome.get("claims", []):
                        rag_claim_records.append(_claim_to_record(scenario, claim))
            except Exception as e:
                logger.error(f"{name} failed for {scenario}: {e}")

        _save_results(results, output_dir)
        _save_claim_records(claim_records, output_dir)
        if with_rag_baseline:
            _save_claim_records(rag_claim_records, output_dir, prefix="rag_claims")

    print(f"\n{'=' * 70}\nEVALUATION COMPLETE: {len(results)} result records across {len(scenarios)} scenarios.\n{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the VERIFY-FIN evaluation harness.")
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Resume: re-run only (scenario, condition) pairs missing from the latest results file.",
    )
    parser.add_argument(
        "--with-rag-baseline",
        action="store_true",
        help="Add the Single-Agent + RAG condition (one LLM call given the SAME retrieved evidence, no verification). "
             "Combine with --fill-missing to add it to an existing run without redoing anything else.",
    )
    parser.add_argument(
        "--redo-failed",
        action="store_true",
        help="With --fill-missing: also re-run baseline rows whose claim extraction FAILED (e.g. an empty report).",
    )
    args = parser.parse_args()
    main(fill_missing=args.fill_missing, with_rag_baseline=args.with_rag_baseline, redo_failed=args.redo_failed)
