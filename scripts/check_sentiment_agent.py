"""
Manual, live integration check for the Sentiment Agent: uses your real
Groq API key and real, current news headlines to produce an actual
sentiment ClaimObject.

Usage (from the project root, with venv activated):
    python scripts/check_sentiment_agent.py
"""

from verifyfin.agents.sentiment_agent import SentimentAgent


def main() -> None:
    print("Running the Sentiment Agent on AAPL (real Groq + real news headlines)...")
    agent = SentimentAgent()

    claims = agent.analyze("AAPL", "2026-Q1", company_name="Apple Inc.")

    if not claims:
        print("\nNo sentiment claim was produced (no news found, or the LLM's response couldn't be parsed).")
        return

    claim = claims[0]
    print(f"\nClaim: {claim.claim_text}")
    print(f"Category: {claim.claim_category.value if hasattr(claim.claim_category, 'value') else claim.claim_category}")
    print(f"Confidence: {claim.confidence_score}")
    print(f"Source type: {claim.source_reference.source_type}")
    print(f"Source link: {claim.source_reference.document_id}")
    print(f"Supporting headlines: {claim.source_reference.excerpt}")

    print("\nSuccess: Sentiment Agent produced a real, source-cited claim end-to-end.")


if __name__ == "__main__":
    main()
