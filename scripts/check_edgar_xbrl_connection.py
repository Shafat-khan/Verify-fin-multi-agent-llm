"""
Manual, live connectivity check against the real SEC EDGAR XBRL
Company Facts API.

Usage (from the project root, with venv activated):
    python scripts/check_edgar_xbrl_connection.py
"""

from verifyfin.data_pipeline.edgar_xbrl_client import EdgarXbrlClient


def main() -> None:
    print("Fetching live XBRL company facts from SEC EDGAR for AAPL...")
    client = EdgarXbrlClient()

    fundamentals = client.get_latest_fundamentals("AAPL")

    print("\n--- AAPL fundamentals (from SEC XBRL, official & audited) ---")
    for concept, entry in fundamentals.items():
        if concept == "ticker":
            continue
        if entry is None:
            print(f"{concept:>20}: (not reported under any known tag)")
        else:
            print(
                f"{concept:>20}: {entry['value']:>18,} {entry['unit']} "
                f"(as of {entry['period_end']}, {entry['form']}, accn {entry['accession_number']})"
            )
    print("----------------------------------------------------------------\n")
    print("Success: SEC EDGAR XBRL access is working correctly.")


if __name__ == "__main__":
    main()
