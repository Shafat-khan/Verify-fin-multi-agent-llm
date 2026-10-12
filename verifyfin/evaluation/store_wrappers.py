"""
verifyfin.evaluation.store_wrappers
======================================
Thin wrappers around VectorStore for the ablation experiments. Both expose
the same query() interface the Verification Agent uses, so they can be
passed as `vector_store=` unchanged.

CachingStore
    Memoises query() results. Many verifier variants issue IDENTICAL
    first-stage retrievals (the variants differ in what they do AFTER
    retrieval), so caching makes an 8-variant ablation cost little more
    than one variant.

FilteredStore
    Emulates a smaller knowledge base by keeping only chunks whose
    `source_type` metadata is in an allowed set (e.g. filing text only,
    or filing + FRED without news). It over-fetches and filters after the
    fact, because ChromaDB's `where` filter is already used for the ticker.
    The knowledge-base ablation therefore needs no re-indexing.
"""

from __future__ import annotations

from typing import Iterable, Optional


class CachingStore:
    def __init__(self, store):
        self._store = store
        self._cache: dict = {}
        self.hits = 0
        self.misses = 0

    def query(self, query_text: str, ticker: Optional[str] = None, top_k: int = 5,
              where_document_contains: Optional[str] = None) -> list[dict]:
        key = (query_text, ticker, top_k, where_document_contains)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        result = self._store.query(query_text, ticker=ticker, top_k=top_k, where_document_contains=where_document_contains)
        self._cache[key] = result
        return result

    def __getattr__(self, name):          # count(), get_stats(), ...
        return getattr(self._store, name)


class FilteredStore:
    def __init__(self, store, allowed_source_types: Iterable[str], overfetch: int = 8):
        self._store = store
        self.allowed = set(allowed_source_types)
        self.overfetch = overfetch

    def query(self, query_text: str, ticker: Optional[str] = None, top_k: int = 5,
              where_document_contains: Optional[str] = None) -> list[dict]:
        raw = self._store.query(query_text, ticker=ticker, top_k=top_k * self.overfetch,
                                where_document_contains=where_document_contains)
        kept = [r for r in raw if (r.get("metadata") or {}).get("source_type") in self.allowed]
        return kept[:top_k]

    def __getattr__(self, name):
        return getattr(self._store, name)


KB_VARIANTS = {
    "all": None,                                   # full knowledge base
    "filing_only": {"edgar_10q", "edgar_10k"},     # filing text only
    "filing_fred": {"edgar_10q", "edgar_10k", "fred"},   # no news headlines
    "fred_news": {"fred", "news"},                 # no filing text
}


def make_kb_view(store, kb_variant: str):
    allowed = KB_VARIANTS[kb_variant]
    return store if allowed is None else FilteredStore(store, allowed)
