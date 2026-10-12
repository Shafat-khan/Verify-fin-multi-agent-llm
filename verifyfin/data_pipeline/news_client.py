"""
verifyfin.data_pipeline.news_client
======================================
Client for retrieving recent news headlines about a ticker, used by
the Sentiment Agent (Proposal Objective 2.3) to ground sentiment
claims in actual recent news coverage rather than the LLM's own
unsourced impression of "how the market feels" about a company.

Deliberately uses Google News RSS, not Yahoo Finance: Yahoo's
anti-scraping measures caused repeated, well-documented problems
elsewhere in this project (see the now-removed yfinance_client
module's history in fundamental_agent.py's docstring). Google News
RSS is a stable, unauthenticated, free public feed with no comparable
blocking behaviour observed during development.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import feedparser

from verifyfin.utils.io_utils import resolve_path
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_GOOGLE_NEWS_RSS_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_CACHE_TTL_HOURS = 6  # news changes far faster than financial statements or macro data


class NewsUnavailableError(Exception):
    """Raised when no news headlines could be retrieved or parsed for a ticker."""


class NewsClient:
    def __init__(self, config_path: Optional[str] = None):
        self.cache_dir = resolve_path("data/raw/news_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = _CACHE_TTL_HOURS

    # ---------- caching ----------

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}_news.json"

    def _read_cache(self, ticker: str) -> Optional[list[dict]]:
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        cached_at = datetime.fromisoformat(payload["cached_at"])
        if datetime.now(timezone.utc) - cached_at > timedelta(hours=self.cache_ttl_hours):
            return None
        return payload["articles"]

    def _write_cache(self, ticker: str, articles: list[dict]) -> None:
        payload = {"cached_at": datetime.now(timezone.utc).isoformat(), "articles": articles}
        with open(self._cache_path(ticker), "w") as f:
            json.dump(payload, f)

    # ---------- public API ----------

    def get_recent_headlines(
        self, ticker: str, company_name: Optional[str] = None, max_articles: int = 10
    ) -> list[dict]:
        """
        Returns up to max_articles recent news headlines for a ticker,
        each as {"title", "link", "published", "source"}. Cached to
        disk with a short TTL (news changes much faster than financial
        statements, so a long TTL like FRED's 24h would be wrong here).
        """
        cached = self._read_cache(ticker)
        if cached is not None:
            return cached[:max_articles]

        query_text = company_name or ticker
        url = _GOOGLE_NEWS_RSS_TEMPLATE.format(query=quote_plus(f"{query_text} stock"))

        logger.info(f"Fetching news headlines for {ticker}...")
        feed = feedparser.parse(url)

        if feed.bozo and not feed.entries:
            raise NewsUnavailableError(
                f"Failed to retrieve or parse the news feed for {ticker}: {feed.get('bozo_exception')}"
            )

        articles = []
        for entry in feed.entries[:max_articles]:
            source = entry.get("source", {})
            articles.append(
                {
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": source.get("title", "") if isinstance(source, dict) else "",
                }
            )

        if not articles:
            raise NewsUnavailableError(f"No news articles found for {ticker}.")

        self._write_cache(ticker, articles)
        return articles
