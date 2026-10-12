"""
Manual, live connectivity check against the real Groq API.

Unlike tests/test_groq_client.py (which is fully mocked and runs
offline as part of the automated test suite), THIS script makes one
real network call using your actual GROQ_API_KEY from .env. Run it
once after setup to confirm your key works.

Usage (from the project root, with venv activated):
    python scripts/check_groq_connection.py
"""

from verifyfin.llm.groq_client import GroqClient


def main() -> None:
    print("Connecting to Groq using the key in your .env file...")
    client = GroqClient()

    response = client.complete(
        user_prompt="Reply with exactly one short sentence confirming you received this test message.",
        system_prompt="You are a connectivity test assistant.",
    )

    print("\n--- Response from Groq ---")
    print(response)
    print("---------------------------\n")
    print("Success: your Groq API key is working correctly.")
    print(f"(This response is now cached at: {client.cache_dir})")


if __name__ == "__main__":
    main()
