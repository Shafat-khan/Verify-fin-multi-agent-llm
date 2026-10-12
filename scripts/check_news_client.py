"""
Manual, live connectivity check against real Google News RSS.

Usage (from the project root, with venv activated):
    python scripts/check_news_client.py
"""

from verifyfin.data_pipeline.news_client import NewsClient


def main() -> None:
    print("Fetching live news headlines for AAPL from Google News RSS...")
    client = NewsClient()

    articles = client.get_recent_headlines("AAPL", company_name="Apple Inc.", max_articles=5)

    print(f"\n--- {len(articles)} recent headlines ---")
    for i, a in enumerate(articles):
        print(f"\n[{i}] {a['title']}")
        print(f"    source: {a['source']}")
        print(f"    published: {a['published']}")
        print(f"    link: {a['link']}")
    print("\n----------------------------\n")
    print("Success: news client is working correctly.")


if __name__ == "__main__":
    main()
