"""
verifyfin.baselines.single_agent_rag_baseline
=================================================
Single-Agent + RAG baseline (faculty feedback point 4).

THE FLAW THIS FIXES
    The original Single-Agent baseline had NO evidence at all and was then
    checked against a specific knowledge base it never saw. Its 100%
    unsupported rate therefore mostly measured "no access to evidence", not
    "no verification". It could not isolate what VERIFY-FIN's verification
    layer contributes.

THE CONTROL
    One LLM call, given the SAME retrieved evidence from the SAME scenario
    knowledge base that VERIFY-FIN's Verification Agent checks against, and
    told to use only that evidence. It has:
        + evidence access   (the same passages)
        - no structured claims, no source references
        - no claim-level verification, no Debate, no gating
    The three single-condition comparisons now isolate three different things:
        Single-Agent (no evidence)  vs  Single-Agent + RAG   -> value of EVIDENCE ACCESS
        Single-Agent + RAG          vs  VERIFY-FIN           -> value of the VERIFICATION PIPELINE
                                                                (structured claims + verification + debate)
        Unverified Multi-Agent      vs  VERIFY-FIN           -> value of VERIFICATION alone

    Retrieval is deliberately simple and generic (one fixed set of queries for every
    company, top-k passages each) so it is not tailored to the verifier.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from verifyfin.evaluation.run_metrics import RunMetrics, collect_run_metrics
from verifyfin.llm.groq_client import GroqClient
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

RETRIEVAL_QUERIES = [
    "total revenue net sales for the quarter",
    "net income",
    "basic earnings per share",
    "diluted earnings per share",
    "total assets",
    "total liabilities",
    "total shareholders equity stockholders equity",
    "operating income",
    "federal funds rate",
    "consumer price index inflation",
    "unemployment rate",
    "gross domestic product GDP",
    "recent news headlines about the company",
]

# The model is a REASONING model: hidden reasoning tokens count against max_tokens, so a small budget can
# be used up before any visible text is written (the first run returned an EMPTY report for 4 of 14
# scenarios at max_tokens=700). Use a generous budget and retry once with a larger one if still empty.
_MAX_TOKENS = 2500
_RETRY_MAX_TOKENS = 5000

_SYSTEM_PROMPT = (
    "You are a financial analyst. Write a concise research summary (150-300 words) about the given "
    "company for the given period, covering its financial performance, the macroeconomic environment "
    "and market sentiment. Use ONLY the numbered evidence passages provided. Every number you state "
    "must appear in the evidence. If the evidence does not support a statement, leave it out. "
    "Do not use any outside knowledge."
)


class SingleAgentRagBaseline:
    def __init__(self, config_path: Optional[str] = None, llm_client: Optional[GroqClient] = None,
                 cache_namespace: Optional[str] = None, top_k_per_query: int = 2, max_passage_chars: int = 700):
        self.config_path = config_path
        self._llm = llm_client if llm_client is not None else GroqClient(config_path, cache_namespace=cache_namespace)
        self.top_k_per_query = top_k_per_query
        self.max_passage_chars = max_passage_chars
        self.last_run_metrics: Optional[RunMetrics] = None
        self.last_context: list[dict] = []
        self.last_report_empty: bool = False
        self.last_attempts: int = 0

    def retrieve_context(self, vector_store, ticker: str) -> list[dict]:
        """The fixed generic queries, top-k passages each, de-duplicated, order preserved."""
        seen, passages = set(), []
        for q in RETRIEVAL_QUERIES:
            for hit in vector_store.query(q, ticker=ticker, top_k=self.top_k_per_query):
                text = hit["text"].strip()
                if text and text not in seen:
                    seen.add(text)
                    passages.append(hit)
        return passages

    def _format_context(self, passages: list[dict]) -> str:
        lines = []
        for i, p in enumerate(passages, 1):
            text = " ".join(p["text"].split())[: self.max_passage_chars]
            kind = (p.get("metadata") or {}).get("source_type", "source")
            lines.append(f"[{i}] ({kind}) {text}")
        return "\n".join(lines)

    def run(self, ticker: str, time_period: str, vector_store, run_id: Optional[str] = None,
            period_end: Optional[str] = None) -> str:
        run_id = run_id or str(uuid.uuid4())
        logger.info(f"[SingleAgentRAG] Generating evidence-grounded report for {ticker} ({time_period})...")
        self._llm.reset_stats()
        if hasattr(vector_store, "reset_stats"):
            vector_store.reset_stats()
        start = time.perf_counter()

        passages = self.retrieve_context(vector_store, ticker)
        self.last_context = passages
        period = f"{time_period}" + (f" (period ended {period_end})" if period_end else "")
        user_prompt = (f"Company: {ticker.upper()}\nPeriod: {period}\n\nEvidence passages:\n"
                       f"{self._format_context(passages)}\n\nWrite the research summary now.")
        report = ""
        self.last_attempts = 0
        for budget in (_MAX_TOKENS, _RETRY_MAX_TOKENS):
            self.last_attempts += 1
            report = (self._llm.complete(user_prompt=user_prompt, system_prompt=_SYSTEM_PROMPT, max_tokens=budget) or "").strip()
            if report:
                break
            logger.warning(f"[SingleAgentRAG] EMPTY report for {ticker} ({time_period}) at max_tokens={budget}.")
        self.last_report_empty = not report
        if self.last_report_empty:
            logger.error(f"[SingleAgentRAG] still EMPTY after {self.last_attempts} attempts for {ticker} ({time_period}); "
                         f"this scenario will have no claims and must be reported as a failure, not as a result.")

        self.last_run_metrics = collect_run_metrics(
            run_id=run_id, ticker=ticker.upper(), time_period=time_period, condition="single_agent_rag",
            runtime_seconds=time.perf_counter() - start, groq_client=self._llm, vector_store=vector_store)
        return report
