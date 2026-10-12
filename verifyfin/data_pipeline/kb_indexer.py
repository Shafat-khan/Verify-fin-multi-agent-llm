"""
verifyfin.data_pipeline.kb_indexer
=====================================
Extends the VectorStore knowledge base with FRED macro time-series
data and recent news headlines, so Macro Agent and Sentiment Agent
claims become genuinely verifiable via the same RAG-based Verification
Agent used for Fundamental Agent claims.

Why this exists (real limitation found during a full pipeline run):
the knowledge base previously contained only SEC filing text, so every
macro and sentiment claim was routed to BLOCKED regardless of whether
it was actually correct -- there was simply nothing in the knowledge
base for a Fed Funds Rate or a news headline to match against.

Indexes INDEPENDENT source content, not each claim's own excerpt --
this matters methodologically:
    - For FRED: the FULL historical observation series (not just the
      single "latest value" MacroAgent's claim cites), so verification
      genuinely has to find the right observation among many, rather
      than trivially matching a copy of what the claim already said.
    - For news: the actual headline TITLE TEXT of each retrieved
      article, independent of how SentimentAgent later phrases its
      claim about them.
    Indexing a claim's own excerpt instead would make "verification"
    circular -- the claim would always find a perfect match because
    it would literally be searching for a copy of itself, which
    verifies nothing. This mirrors how EDGAR filing text works: the
    knowledge base is the independently-fetched source document, not
    a record of what an agent already claimed about it.
"""

from __future__ import annotations

from typing import Optional

from verifyfin.data_pipeline.chunker import Chunk
from verifyfin.data_pipeline.fred_client import FredClient
from verifyfin.data_pipeline.news_client import NewsClient, NewsUnavailableError
from verifyfin.data_pipeline.vector_store import VectorStore
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)


def index_macro_data(store: VectorStore, ticker: str, fred_client: Optional[FredClient] = None,
                     as_of: Optional[str] = None) -> int:
    """
    Fetches the full historical series for every configured FRED
    indicator and indexes each observation as its own small chunk,
    tagged to `ticker` so the existing ticker-filtered
    VerificationAgent.verify_claim() can find them with no changes to
    that agent. Returns the number of chunks added.

    as_of ("YYYY-MM-DD", typically the scenario's period end): observations
    dated AFTER it are not indexed, so a historical scenario's knowledge base
    holds no data from the future (no look-ahead). None = index everything.
    """
    fred = fred_client or FredClient()
    chunks: list[Chunk] = []

    for series_id in fred.default_series:
        try:
            series = fred.get_series(series_id)
        except Exception as e:
            logger.warning(f"Skipping FRED series {series_id} for indexing: {e}")
            continue

        for date, value in zip(series["dates"], series["values"]):
            if as_of is not None and date > as_of:
                continue
            chunks.append(
                Chunk(
                    text=f"{series_id} observation: {value} on {date}",
                    chunk_index=len(chunks),
                    source_document_id=series_id,
                    source_type="fred",
                )
            )

    added = store.add_chunks(chunks, ticker=ticker)
    logger.info(f"Indexed {added} FRED observation chunks for {ticker}.")
    return added


def index_news_headlines(
    store: VectorStore,
    ticker: str,
    company_name: Optional[str] = None,
    news_client: Optional[NewsClient] = None,
) -> int:
    """
    Fetches recent news headlines and indexes each headline's title as
    its own chunk, tagged to `ticker`. Returns the number of chunks
    added (0, not an error, if no news is available -- matching the
    graceful-degradation pattern used elsewhere in the pipeline).
    """
    news = news_client or NewsClient()
    try:
        articles = news.get_recent_headlines(ticker, company_name=company_name)
    except NewsUnavailableError as e:
        logger.warning(f"Skipping news indexing for {ticker}: {e}")
        return 0

    chunks = [
        Chunk(
            text=a["title"],
            chunk_index=i,
            source_document_id=a["link"] or f"{ticker}-headline-{i}",
            source_type="news",
        )
        for i, a in enumerate(articles)
    ]

    added = store.add_chunks(chunks, ticker=ticker)
    logger.info(f"Indexed {added} news headline chunks for {ticker}.")
    return added
