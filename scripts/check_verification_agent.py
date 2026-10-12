"""
Manual, live end-to-end integration check: builds a small real
knowledge base from an AAPL 10-Q, produces real claims via the
Fundamental Agent, then verifies each claim against the knowledge
base -- computing real grounding scores and verdicts.

This is the first time the actual core mechanism of VERIFY-FIN (claim
generation -> mid-pipeline grounding check) runs end-to-end with real
data. Expect most/all claims to come back GROUNDED, since they were
generated directly from the same filing that's indexed in the
knowledge base -- this is a sanity check that the pipeline works
correctly, not yet the full evaluation (Proposal §2.5's 15-scenario,
3-condition comparison comes later).

Usage (from the project root, with venv activated):
    python scripts/check_verification_agent.py
"""

from verifyfin.agents.fundamental_agent import FundamentalAgent
from verifyfin.agents.verification_agent import VerificationAgent
from verifyfin.data_pipeline.chunker import Chunker
from verifyfin.data_pipeline.edgar_client import EdgarClient
from verifyfin.data_pipeline.vector_store import VectorStore


def _verdict_str(claim) -> str:
    return claim.verdict.value if hasattr(claim.verdict, "value") else claim.verdict


def main() -> None:
    ticker = "AAPL"

    print(f"Building knowledge base for {ticker}...")
    edgar = EdgarClient()
    filings = edgar.get_filings_for_ticker(ticker, filing_types=["10-Q"], max_filings=1)
    filing = filings[0]

    chunker = Chunker()
    chunks = chunker.chunk_text(
        filing["clean_text"], source_document_id=filing["accession_number"], source_type="edgar_10q"
    )

    store = VectorStore()
    store.add_chunks(chunks, ticker=ticker)
    print(f"Knowledge base contains {store.count()} chunks.\n")

    print("Producing claims via the Fundamental Agent (real Groq call)...")
    fundamental_agent = FundamentalAgent()
    claims = fundamental_agent.analyze(ticker, "2026-Q1")
    print(f"Produced {len(claims)} claims.\n")

    print("Verifying claims against the knowledge base...")
    # Reuse the same VectorStore instance -- avoids reloading the embedding model a second time.
    verification_agent = VerificationAgent(vector_store=store)
    verified_claims = verification_agent.verify_claims(claims)

    print("\n--- Verification results ---")
    for c in verified_claims:
        print(f"\n[{_verdict_str(c).upper()}] grounding_score={c.grounding_score}")
        print(f"  Claim:    {c.claim_text}")
        evidence_preview = (c.retrieved_evidence or "")[:150]
        print(f"  Evidence: {evidence_preview}...")

    grounded = sum(1 for c in verified_claims if _verdict_str(c) == "grounded")
    print(f"\n-----------------------------\n")
    print(f"{grounded}/{len(verified_claims)} claims came back GROUNDED.")
    print("Success: Verification Agent computed real grounding scores end-to-end.")


if __name__ == "__main__":
    main()
