"""
Manual, live INTEGRATION check: fetches one real EDGAR filing, chunks
it, embeds + stores it in ChromaDB, then runs a retrieval query against
it. This is an end-to-end preview of the knowledge-base build
(Proposal Objective 2.3/2.4) and the closest thing so far to a real
"claim grounding" lookup.

NOTE: the first run downloads the sentence-transformers embedding
model (~80MB) -- this can take a minute depending on your connection,
but only happens once (it's cached locally after that).

Usage (from the project root, with venv activated):
    python scripts/check_vector_store.py
"""

from verifyfin.data_pipeline.chunker import Chunker
from verifyfin.data_pipeline.edgar_client import EdgarClient
from verifyfin.data_pipeline.vector_store import VectorStore


def main() -> None:
    print("Fetching one real AAPL 10-Q filing from SEC EDGAR...")
    edgar = EdgarClient()
    filings = edgar.get_filings_for_ticker("AAPL", filing_types=["10-Q"], max_filings=1)
    filing = filings[0]
    print(f"Retrieved {len(filing['clean_text']):,} characters of filing text "
          f"({filing['form']}, filed {filing['filing_date']}).")

    print("\nChunking the filing text...")
    chunker = Chunker()
    chunks = chunker.chunk_text(
        filing["clean_text"],
        source_document_id=filing["accession_number"],
        source_type="edgar_10q",
    )
    print(f"Produced {len(chunks)} chunks.")

    print("\nLoading embedding model and building the vector store "
          "(first run downloads the model, ~80MB)...")
    store = VectorStore()
    store.add_chunks(chunks, ticker="AAPL")
    print(f"Vector store now contains {store.count()} total chunks.")

    query = "What was Apple's revenue in the most recent quarter?"
    print(f"\nQuerying: '{query}'")
    results = store.query(query, ticker="AAPL", top_k=3)

    print("\n--- Top matching chunks (this is what the Verification Agent will see) ---")
    for i, r in enumerate(results, 1):
        preview = r["text"][:250] + ("..." if len(r["text"]) > 250 else "")
        print(f"\n[{i}] similarity_score={r['similarity_score']}")
        print(preview)
    print("\n-----------------------------------------------------------------------\n")
    print("Success: chunking + embedding + retrieval pipeline is working end-to-end.")


if __name__ == "__main__":
    main()
