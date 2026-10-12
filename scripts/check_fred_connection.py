"""
Manual, live connectivity check against the real FRED API using your
FRED_API_KEY from .env.

Usage (from the project root, with venv activated):
    python scripts/check_fred_connection.py
"""

from verifyfin.data_pipeline.fred_client import FredClient


def main() -> None:
    print("Connecting to FRED using the key in your .env file...")
    client = FredClient()

    snapshot = client.get_macro_snapshot()
    print("\n--- Macro snapshot ---")
    for series_id, data in snapshot.items():
        print(f"{series_id:>10}: {data['latest_value']}  (as of {data['latest_date']})")
    print("----------------------\n")
    print("Success: your FRED API key is working correctly.")


if __name__ == "__main__":
    main()
