"""
Manual, live integration check for the Fundamental Agent: uses your
real Groq API key and real SEC EDGAR XBRL data to produce actual
ClaimObjects. yfinance is attempted best-effort and will be skipped
gracefully if Yahoo is still blocking your network.

Usage (from the project root, with venv activated):
    python scripts/check_fundamental_agent.py
"""

import json

from verifyfin.agents.fundamental_agent import FundamentalAgent


def main() -> None:
    print("Running the Fundamental Agent on AAPL (real Groq + real SEC data)...")
    agent = FundamentalAgent()

    claims = agent.analyze("AAPL", "2026-Q1")

    print(f"\nProduced {len(claims)} ClaimObjects:\n")
    for i, claim in enumerate(claims, 1):
        print(f"[{i}] {claim.claim_text}")
        print(f"    category: {claim.claim_category.value if hasattr(claim.claim_category, 'value') else claim.claim_category}")
        print(f"    source:   {claim.source_reference.source_type} / {claim.source_reference.document_id}")
        print(f"    excerpt:  {claim.source_reference.excerpt}")
        print(f"    confidence: {claim.confidence_score}")
        print()

    print("Success: Fundamental Agent produced real, source-cited claims end-to-end.")


if __name__ == "__main__":
    main()
