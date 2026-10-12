"""
verifyfin.baselines.unverified_multiagent_baseline
======================================================
Unverified Multi-Agent Baseline (Proposal §2.5 evaluation design,
Condition 2): same analyst agents and real data as VERIFY-FIN, but
Verification and Debate are skipped entirely.

cache_namespace: passed through to the internal GroqClient so this
condition's LLM cost is measured independently of the other two
conditions -- see groq_client module docstring for the real confound
this fixes (this condition's tokens were previously appearing HIGHER
than the strictly more expensive VERIFY-FIN condition, purely because
of cross-condition cache sharing within the same scenario).
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from verifyfin.agents.fundamental_agent import FundamentalAgent
from verifyfin.agents.macro_agent import MacroAgent
from verifyfin.agents.sentiment_agent import SentimentAgent
from verifyfin.agents.synthesis_agent import SynthesisAgent
from verifyfin.evaluation.run_metrics import RunMetrics, collect_run_metrics
from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.state import AgentState, PipelineStage
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)


class UnverifiedMultiAgentBaseline:
    def __init__(self, config_path: Optional[str] = None, cache_namespace: Optional[str] = None):
        self.config_path = config_path
        self._llm = GroqClient(config_path, cache_namespace=cache_namespace)

        self._fundamental_agent = FundamentalAgent(config_path)
        self._fundamental_agent._llm = self._llm
        self._macro_agent = MacroAgent(config_path, llm_client=self._llm)
        self._sentiment_agent = SentimentAgent(config_path, llm_client=self._llm)
        self._synthesis_agent = SynthesisAgent(config_path, llm_client=self._llm)

        self.last_run_metrics: Optional[RunMetrics] = None

    def run(
        self,
        ticker: str,
        time_period: str,
        run_id: Optional[str] = None,
        target_period_end: Optional[str] = None,
        target_macro_date: Optional[str] = None,
    ) -> AgentState:
        run_id = run_id or str(uuid.uuid4())
        state = AgentState(
            run_id=run_id,
            ticker=ticker.upper(),
            time_period=time_period,
            condition="unverified_multiagent",
            target_period_end=target_period_end,
            target_macro_date=target_macro_date,
            stage=PipelineStage.INITIALISED,
        )

        self._llm.reset_stats()
        start = time.perf_counter()

        logger.info(f"[UnverifiedBaseline] Running analyst agents for {ticker} ({time_period})...")

        try:
            state.raw_claims.extend(
                self._fundamental_agent.analyze(ticker, time_period, target_period_end=target_period_end)
            )
        except Exception as e:
            state.error_log.append(f"Fundamental Agent failed: {e}")
            logger.warning(f"Fundamental Agent failed: {e}")

        try:
            state.raw_claims.extend(
                self._macro_agent.analyze(ticker, time_period, target_date=target_macro_date)
            )
        except Exception as e:
            state.error_log.append(f"Macro Agent failed: {e}")
            logger.warning(f"Macro Agent failed: {e}")

        try:
            state.raw_claims.extend(self._sentiment_agent.analyze(ticker, time_period))
        except Exception as e:
            state.error_log.append(f"Sentiment Agent failed: {e}")
            logger.warning(f"Sentiment Agent failed: {e}")

        logger.info(
            f"[UnverifiedBaseline] {len(state.raw_claims)} claims produced; "
            f"skipping verification/debate by design (this is the baseline condition)."
        )

        state.stage = PipelineStage.SYNTHESIS_RUNNING
        state.final_report = self._synthesis_agent.synthesize(state, require_verification=False)
        state.stage = PipelineStage.COMPLETE

        runtime_seconds = time.perf_counter() - start
        self.last_run_metrics = collect_run_metrics(
            run_id=run_id,
            ticker=ticker.upper(),
            time_period=time_period,
            condition="unverified_multiagent",
            runtime_seconds=runtime_seconds,
            claims=state.raw_claims,
            groq_client=self._llm,
        )

        return state
