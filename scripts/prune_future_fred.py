"""
Remove FRED observations dated AFTER each scenario's period end from the EXISTING knowledge bases (no look-ahead).

WHY
    fred_client.get_series() fetched every observation up to today and index_macro_data() indexed them all, so a
    knowledge base for a quarter ending 28 March 2026 also held observations dated April-September 2026. Claims never
    needed them, but a knowledge base that contains the future is not an "as of the period end" knowledge base, and
    the verifier can match a claim to a chunk from after the period. New knowledge bases are now built with a cutoff
    (index_macro_data(as_of=period_end)); this script applies the same cutoff to the ones you already have, without
    re-fetching anything. Filing chunks, news chunks and the vector store itself are untouched.

USAGE (project root, venv active)
    python scripts/prune_future_fred.py            # DRY RUN: only counts what would be removed
    python scripts/prune_future_fred.py --apply    # actually delete those chunks
"""

from __future__ import annotations

import argparse
import glob
import json
from typing import Callable, Optional

from verifyfin.agents.evidence_alignment import parse_fred_chunk


def future_fred_ids(ids: list[str], documents: list[str], period_end: str) -> list[str]:
    """Ids of FRED observation chunks dated strictly after period_end."""
    out = []
    for chunk_id, doc in zip(ids, documents):
        parsed = parse_fred_chunk(doc)
        if parsed and parsed[2] > period_end:
            out.append(chunk_id)
    return out


def prune_collection(collection, period_end: str, apply: bool) -> tuple[int, int]:
    """Returns (FRED chunks in the collection, FRED chunks dated after period_end). Deletes them only if apply."""
    got = collection.get(where={"source_type": "fred"}, include=["documents"])
    ids, docs = got.get("ids", []), got.get("documents", [])
    future = future_fred_ids(ids, docs, period_end)
    if apply and future:
        collection.delete(ids=future)
    return len(ids), len(future)


def _open_client():
    import chromadb

    from verifyfin.utils.io_utils import load_config, resolve_path

    cfg = load_config(None)["vector_store"]
    return chromadb.PersistentClient(path=str(resolve_path(cfg.get("persist_dir", "./data/chroma_store"))),
                                     settings=chromadb.config.Settings(anonymized_telemetry=False))


def main(argv=None, client_factory: Optional[Callable] = None) -> None:
    ap = argparse.ArgumentParser(description="Remove FRED chunks dated after each scenario's period end.")
    ap.add_argument("--results")
    ap.add_argument("--apply", action="store_true", help="delete (default is a dry run)")
    args = ap.parse_args(argv)
    results = args.results or (sorted(glob.glob("outputs/results/evaluation_results_*.json")) or [None])[-1]
    if not results:
        raise SystemExit("No evaluation_results_*.json found; pass --results.")
    scen = sorted({(r["ticker"], r["period_end"]) for r in json.load(open(results)) if r.get("period_end")})
    client = (client_factory or _open_client)()
    total_f = total_x = 0
    for ticker, period_end in scen:
        name = f"eval_{ticker}_{period_end}"
        try:
            coll = client.get_collection(name)
        except Exception as e:
            print(f"  {name}: not found ({e}); skipped")
            continue
        n_f, n_x = prune_collection(coll, period_end, args.apply)
        total_f += n_f
        total_x += n_x
        print(f"  {name}: {n_f} FRED chunks, {n_x} dated after {period_end} -> {'DELETED' if args.apply else 'would delete'}")
    print(f"\nTotal: {total_x} of {total_f} FRED chunks are from after their scenario's period end."
          + ("" if args.apply else "  (dry run: nothing changed; add --apply to delete them)"))


if __name__ == "__main__":
    main()
