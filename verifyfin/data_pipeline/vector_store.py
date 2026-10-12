"""
verifyfin.data_pipeline.vector_store
=======================================
ChromaDB-backed vector store wrapping sentence-transformers embeddings.

This is the knowledge base the Verification Agent retrieves against
when computing a claim's grounding score.

Chroma is configured to use cosine distance explicitly, so the
similarity scores returned here are directly comparable to the
grounded_threshold / challenged_threshold values in config.yaml.

Retrieval instrumentation: every query() call is timed and counted.
get_stats()/reset_stats() expose this for the evaluation harness.

collection_name override (added for the evaluation harness): by
default, the collection name comes from config.yaml, and chunks are
scoped only by a `ticker` metadata filter. For a multi-period
evaluation (5 tickers x 3 quarters), that's not enough isolation --
two different quarters for the SAME ticker would share the `ticker`
filter and contaminate each other's retrieval. Passing a distinct
collection_name per (ticker, quarter) scenario gives each scenario a
fully separate knowledge base within the same underlying Chroma store.
"""

from __future__ import annotations

import time
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from verifyfin.data_pipeline.chunker import Chunk
from verifyfin.utils.io_utils import load_config, resolve_path
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, config_path: Optional[str] = None, collection_name: Optional[str] = None):
        self.config = load_config(config_path)
        vs_cfg = self.config["vector_store"]
        emb_cfg = self.config["embedding"]

        persist_dir = str(resolve_path(vs_cfg.get("persist_dir", "./data/chroma_store")))
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        resolved_collection_name = collection_name or vs_cfg.get("collection_name", "verify_fin_knowledge_base")
        self._collection = self._client.get_or_create_collection(
            name=resolved_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"Loading embedding model {emb_cfg['model']}...")
        self._embedder = SentenceTransformer(emb_cfg["model"], device=emb_cfg.get("device", "cpu"))

        self.query_count = 0
        self.total_query_time_seconds = 0.0

    def add_chunks(self, chunks: list[Chunk], ticker: str) -> int:
        if not chunks:
            return 0

        ids = [f"{ticker}_{c.source_document_id}_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "ticker": ticker,
                "source_document_id": c.source_document_id,
                "source_type": c.source_type,
                "section_hint": c.section_hint or "",
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        embeddings = self._embedder.encode(documents, show_progress_bar=False).tolist()

        self._collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.info(f"Added/updated {len(chunks)} chunks for {ticker} in the vector store.")
        return len(chunks)

    def query(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        top_k: int = 5,
        where_document_contains: Optional[str] = None,
    ) -> list[dict]:
        start = time.perf_counter()
        try:
            query_embedding = self._embedder.encode([query_text], show_progress_bar=False).tolist()

            where_filter = {"ticker": ticker} if ticker else None
            where_document_filter = {"$contains": where_document_contains} if where_document_contains else None

            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=where_filter,
                where_document=where_document_filter,
            )
        finally:
            self.query_count += 1
            self.total_query_time_seconds += time.perf_counter() - start

        matches = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            similarity = 1.0 - dist
            matches.append({"text": doc, "metadata": meta, "similarity_score": round(similarity, 4)})

        return matches

    def count(self) -> int:
        return self._collection.count()

    def get_stats(self) -> dict:
        return {
            "query_count": self.query_count,
            "total_query_time_seconds": round(self.total_query_time_seconds, 4),
        }

    def reset_stats(self) -> None:
        self.query_count = 0
        self.total_query_time_seconds = 0.0
