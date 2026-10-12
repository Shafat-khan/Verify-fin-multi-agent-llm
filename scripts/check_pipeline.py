"""
Manual, live end-to-end integration check for the FULL VERIFY-FIN
pipeline: Orchestrator -> Fundamental/Macro/Sentiment agents ->
Verification -> Debate -> Synthesis, wired together via LangGraph.

Knowledge base now includes THREE independent source types (extended
after the first full pipeline run showed macro/sentiment claims were
architecturally unable to verify against a filing-only knowledge
base):
    1. EDGAR filing text (for Fundamental Agent claims)
    2. FRED historical time series (for Macro Agent claims)
    3. Recent news headlines (for Sentiment Agent claims)

NOTE: this makes several real Groq API calls and real calls to SEC
EDGAR, FRED, and Google News. Expect this to take 30-90 seconds.

Usage (from the project root, with venv activated):
    python scripts/check_pipeline.py
"""

from verifyfin.data_pipeline.chunker import Chunker
from verifyfin.data_pipeline.edgar_client import EdgarClient
from verifyfin.data_pipeline.kb_indexer import index_macro_data, index_news_headlines
from verifyfin.data_pipeline.vector_store import VectorStore
from verifyfin.graph.pipeline import VerifyFinPipeline


def _verdict_str(claim) -> str:
    return claim.verdict.value if hasattr(claim.verdict, "value") else claim.verdict


def main() -> None:
    ticker = "AAPL"
    time_period = "2026-Q3"

    store = VectorStore()

    print(f"Indexing EDGAR filing text for {ticker}...")
    edgar = EdgarClient()
    filings = edgar.get_filings_for_ticker(ticker, filing_types=["10-Q"], max_filings=1)
    filing = filings[0]
    chunker = Chunker()
    chunks = chunker.chunk_text(
        filing["clean_text"], source_document_id=filing["accession_number"], source_type="edgar_10q"
    )
    store.add_chunks(chunks, ticker=ticker)

    print(f"Indexing FRED macro data for {ticker}...")
    index_macro_data(store, ticker)

    print(f"Indexing news headlines for {ticker}...")
    index_news_headlines(store, ticker, company_name="Apple Inc.")

    print(f"\nKnowledge base contains {store.count()} chunks total.\n")

    print("Running the full VERIFY-FIN pipeline (this makes several real API calls)...\n")
    pipeline = VerifyFinPipeline(vector_store=store)
    final_state = pipeline.run(ticker, time_period)

    print("\n" + "=" * 70)
    print(f"PIPELINE RUN COMPLETE -- run_id={final_state.run_id}")
    print("=" * 70)

    print(f"\nTotal claims produced: {len(final_state.raw_claims)}")
    print(f"Errors logged: {final_state.error_log or 'none'}")

    all_claims = final_state.debated_claims if final_state.debated_claims else final_state.verified_claims
    print(f"\n--- All claims and their verdicts ---")
    for c in all_claims:
        usable = c.is_usable_in_final_report()
        category = c.claim_category.value if hasattr(c.claim_category, "value") else c.claim_category
        print(f"[{_verdict_str(c).upper()}, usable={usable}, category={category}] {c.claim_text}")

    usable_count = sum(1 for c in all_claims if c.is_usable_in_final_report())
    print(f"\n{usable_count}/{len(all_claims)} claims usable in final report.")

    print("\n--- FINAL SYNTHESIZED REPORT ---\n")
    print(final_state.final_report)
    print("\n---------------------------------\n")

    print(f"Overall hallucination rate: {final_state.hallucination_rate():.2%}")

    # Break down by category, since an aggregate rate can obscure very
    # different reliability across claim types (see kb_indexer module
    # docstring / development history).
    print("\nBreakdown by claim category:")
    categories = set(
        c.claim_category.value if hasattr(c.claim_category, "value") else c.claim_category for c in all_claims
    )
    for cat in sorted(categories):
        cat_claims = [
            c for c in all_claims
            if (c.claim_category.value if hasattr(c.claim_category, "value") else c.claim_category) == cat
        ]
        cat_usable = sum(1 for c in cat_claims if c.is_usable_in_final_report())
        print(f"  {cat}: {cat_usable}/{len(cat_claims)} usable")

    print("\nSuccess: full VERIFY-FIN pipeline ran end-to-end via LangGraph with an extended knowledge base.")


if __name__ == "__main__":
    main()
