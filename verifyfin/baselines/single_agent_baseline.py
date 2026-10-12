"""
verifyfin.baselines.single_agent_baseline
=============================================
Single-Agent Baseline (Proposal §2.5 evaluation design, Condition 1):
one LLM call, no data, no verification.

cache_namespace: passed through to GroqClient so this condition's LLM
cost is measured independently of the other two conditions -- see
groq_client module docstring for the real confound this fixes.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from verifyfin.evaluation.run_metrics import RunMetrics, collect_run_metrics
from verifyfin.llm.groq_client import GroqClient
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial analyst. Write a concise research summary (150-300 words) about "
    "the given company for the given period, covering its financial performance, the "
    "macroeconomic environment, and market sentiment. Use your own knowledge -- you have "
    "no other tools or data available, and no ability to look anything up."
)


class SingleAgentBaseline:
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_client: Optional[GroqClient] = None,
        cache_namespace: Optional[str] = None,
    ):
        self.config_path = config_path
        self._llm = llm_client if llm_client is not None else GroqClient(config_path, cache_namespace=cache_namespace)
        self.last_run_metrics: Optional[RunMetrics] = None

    def run(self, ticker: str, time_period: str, run_id: Optional[str] = None) -> str:
        run_id = run_id or str(uuid.uuid4())
        logger.info(f"[SingleAgentBaseline] Generating ungrounded report for {ticker} ({time_period})...")

        self._llm.reset_stats()
        start = time.perf_counter()

        user_prompt = f"Write a financial research summary about {ticker} for the period {time_period}."
        report = self._llm.complete(user_prompt=user_prompt, system_prompt=_SYSTEM_PROMPT).strip()

        runtime_seconds = time.perf_counter() - start
        self.last_run_metrics = collect_run_metrics(
            run_id=run_id,
            ticker=ticker.upper(),
            time_period=time_period,
            condition="single_agent",
            runtime_seconds=runtime_seconds,
            groq_client=self._llm,
        )

        return report
