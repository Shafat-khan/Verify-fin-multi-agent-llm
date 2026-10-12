"""
Print what the knowledge-base chunks for a scenario ACTUALLY look like (read-only; no API calls, no changes).

Used to design a period-aware match: before requiring that a matched filing chunk "belongs to the claimed
period", we need to see how the period appears in real chunks (header text? month-name date? only the row?).

    python scripts/inspect_chunks.py                       # default scenario AAPL-2026-Q2, 3 queries
    python scripts/inspect_chunks.py --scenario JPM-2026-Q1
Output is written to outputs/results/chunk_samples.txt AND printed.
"""
import argparse
import glob
import json
from pathlib import Path

QUERIES = ["total revenue net sales for the quarter", "net income", "diluted earnings per share",
           "total assets", "federal funds rate observation"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="AAPL-2026-Q2")
    ap.add_argument("--results")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args(argv)
    results = args.results or sorted(glob.glob("outputs/results/evaluation_results_*.json"))[-1]
    scen = {r["time_period"]: (r["ticker"], r["period_end"]) for r in json.load(open(results)) if r.get("period_end")}
    ticker, period_end = scen[args.scenario]

    from verifyfin.data_pipeline.vector_store import VectorStore
    store = VectorStore(collection_name=f"eval_{ticker}_{period_end}")
    lines = [f"Scenario {args.scenario}  ticker={ticker}  period_end={period_end}  chunks in KB={store.count()}", ""]
    for q in QUERIES:
        lines.append(f"=== query: {q}")
        for hit in store.query(q, ticker=ticker, top_k=args.top):
            md = hit.get("metadata") or {}
            lines.append(f"--- score={hit['similarity_score']:.3f} metadata={json.dumps(md, default=str)[:300]}")
            lines.append(hit["text"][:900])
            lines.append("")
    out = "\n".join(lines)
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    Path("outputs/results/chunk_samples.txt").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
