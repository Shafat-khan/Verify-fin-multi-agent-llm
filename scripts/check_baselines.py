"""
Manual, live comparison of all three evaluation conditions (Proposal
§2.5) on the same ticker: Single-Agent baseline, Unverified
Multi-Agent baseline, and full VERIFY-FIN. This is a qualitative
preview, not the actual evaluation harness (which comes next) -- it
runs one ticker once, side by side, so you can see the difference in
report quality and grounding with your own eyes before we build the
full 5-ticker x 3-period statistical evaluation.

NOTE: makes several real API calls across all three conditions.

Usage (from the project root, with venv activated):
    python scripts/check_baselines.py
"""

from verifyfin.baselines.single_agent_baseline import SingleAgentBaseline
from verifyfin.baselines.unverified_multiagent_baseline import UnverifiedMultiAgentBaseline
from verifyfin.data_pipeline.chunker import Chunker
from verifyfin.data_pipeline.edgar_client import EdgarClient
from verifyfin.data_pipeline.kb_indexer import index_macro_data, index_news_headlines
from verifyfin.data_pipeline.vector_store import VectorStore
from verifyfin.graph.pipeline import VerifyFinPipeline


def main() -> None:
    ticker = "AAPL"
    time_period = "2026-Q3"

    print("=" * 70)
    print("CONDITION 1: Single-Agent Baseline (no data, no verification)")
    print("=" * 70)
    single_agent = SingleAgentBaseline()
    single_report = single_agent.run(ticker, time_period)
    print(f"\n{single_report}\n")

    print("=" * 70)
    print("CONDITION 2: Unverified Multi-Agent Baseline (real data, no verification)")
    print("=" * 70)
    unverified = UnverifiedMultiAgentBaseline()
    unverified_state = unverified.run(ticker, time_period)
    print(f"\nClaims produced: {len(unverified_state.raw_claims)} (none verified)")
    print(f"\n{unverified_state.final_report}\n")

    print("=" * 70)
    print("CONDITION 3: Full VERIFY-FIN (real data, full verification + debate)")
    print("=" * 70)
    store = VectorStore()
    edgar = EdgarClient()
    filings = edgar.get_filings_for_ticker(ticker, filing_types=["10-Q"], max_filings=1)
    chunker = Chunker()
    chunks = chunker.chunk_text(
        filings[0]["clean_text"], source_document_id=filings[0]["accession_number"], source_type="edgar_10q"
    )
    store.add_chunks(chunks, ticker=ticker)
    index_macro_data(store, ticker)
    index_news_headlines(store, ticker, company_name="Apple Inc.")

    pipeline = VerifyFinPipeline(vector_store=store)
    verify_fin_state = pipeline.run(ticker, time_period)
    usable = sum(1 for c in verify_fin_state.debated_claims if c.is_usable_in_final_report())
    print(f"\nClaims produced: {len(verify_fin_state.raw_claims)}, {usable} verified as usable")
    print(f"\n{verify_fin_state.final_report}\n")

    print("=" * 70)
    print("Side-by-side comparison complete.")
    print("Note: this is a single qualitative run, not the statistical evaluation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
