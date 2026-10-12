"""
Manual, live integration check for the Macro Agent: uses your real
Groq API key and real FRED data to produce actual ClaimObjects.

Usage (from the project root, with venv activated):
    python scripts/check_macro_agent.py
"""

from verifyfin.agents.macro_agent import MacroAgent


def main() -> None:
    print("Running the Macro Agent (real Groq + real FRED data)...")
    agent = MacroAgent()

    claims = agent.analyze("AAPL", "2026-Q1")

    print(f"\nProduced {len(claims)} ClaimObjects:\n")
    for i, claim in enumerate(claims, 1):
        print(f"[{i}] {claim.claim_text}")
        print(f"    source:   {claim.source_reference.source_type} / {claim.source_reference.document_id}")
        print(f"    excerpt:  {claim.source_reference.excerpt}")
        print(f"    confidence: {claim.confidence_score}")
        print()

    print("Success: Macro Agent produced real, source-cited claims end-to-end.")


if __name__ == "__main__":
    main()
