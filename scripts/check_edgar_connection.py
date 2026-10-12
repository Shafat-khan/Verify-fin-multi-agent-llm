"""
Manual, live connectivity check against the real SEC EDGAR API.

Deliberately lightweight: only resolves a ticker to its CIK and lists
its most recent 10-Q metadata -- it does NOT download a full filing
document, to keep this quick and considerate of SEC's servers. Run the
full pipeline via scripts/build_knowledge_base.py once that exists.

Usage (from the project root, with venv activated):
    python scripts/check_edgar_connection.py
"""

from verifyfin.data_pipeline.edgar_client import EdgarClient


def main() -> None:
    print("Connecting to SEC EDGAR...")
    client = EdgarClient()

    ticker = "AAPL"
    cik = client.get_cik_for_ticker(ticker)
    print(f"\n{ticker} -> CIK {cik}")

    filings = client.get_filings_list(ticker, filing_types=["10-Q"], max_filings=1)
    filing = filings[0]

    print("\n--- Most recent 10-Q metadata ---")
    print(f"Form:            {filing['form']}")
    print(f"Filing date:     {filing['filing_date']}")
    print(f"Report date:     {filing['report_date']}")
    print(f"Accession no.:   {filing['accession_number']}")
    print(f"Primary doc:     {filing['primary_document']}")
    print("----------------------------------\n")
    print("Success: SEC EDGAR access is working correctly.")


if __name__ == "__main__":
    main()
