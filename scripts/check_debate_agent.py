"""
Manual, live end-to-end integration check: runs the full pipeline
(knowledge base -> Fundamental Agent -> Verification Agent -> Debate
Agent) and shows exactly how the Debate Agent resolves any CHALLENGED
claims -- the direct continuation of check_verification_agent.py.

Usage (from the project root, with venv activated):
    python scripts/check_debate_agent.py
"""

from verifyfin.agents.debate_agent import DebateAgent
from verifyfin.agents.fundamental_agent import FundamentalAgent
from verifyfin.agents.verification_agent import VerificationAgent
from verifyfin.data_pipeline.chunker import Chunker
from verifyfin.data_pipeline.edgar_client import EdgarClient
from verifyfin.data_pipeline.vector_store import VectorStore
from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.claim import GroundingVerdict


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

    # Share a single GroqClient across agents -- one disk cache, one rate limiter.
    llm_client = GroqClient()

    print("Producing claims via the Fundamental Agent...")
    fundamental_agent = FundamentalAgent()
    fundamental_agent._llm = llm_client  # reuse the shared client
    claims = fundamental_agent.analyze(ticker, "2026-Q1")
    print(f"Produced {len(claims)} claims.\n")

    print("Verifying claims against the knowledge base...")
    verification_agent = VerificationAgent(vector_store=store)
    verified_claims = verification_agent.verify_claims(claims)

    challenged_before = sum(1 for c in verified_claims if c.verdict == GroundingVerdict.CHALLENGED)
    print(f"{challenged_before} claim(s) came back CHALLENGED -- these go to the Debate Agent.\n")

    print("Running Debate Agent on CHALLENGED claims...")
    debate_agent = DebateAgent(llm_client=llm_client)
    debated_claims = debate_agent.debate_claims(verified_claims)

    print("\n--- Final results after verification + debate ---")
    for c in debated_claims:
        usable = c.is_usable_in_final_report()
        debate_note = f", debate={c.debate_resolution}" if c.debate_resolution else ""
        print(f"\n[{_verdict_str(c).upper()}{debate_note}] usable_in_report={usable}")
        print(f"  Claim: {c.claim_text}")

    usable_count = sum(1 for c in debated_claims if c.is_usable_in_final_report())
    print(f"\n---------------------------------------------------\n")
    print(f"{usable_count}/{len(debated_claims)} claims are usable in the final report.")
    print("Success: full verify -> debate pipeline ran end-to-end.")


if __name__ == "__main__":
    main()
