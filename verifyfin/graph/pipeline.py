"""
verifyfin.graph.pipeline
===========================
LangGraph StateGraph wiring for the full VERIFY-FIN pipeline (Proposal
§2.3.5): Orchestrator -> [Fundamental, Sentiment, Macro] analyst agents
-> Verification Agent -> Debate Agent -> Synthesis Agent.

Node functions each take the shared AgentState and return a dict of
partial state updates; LangGraph merges these automatically.

Note on "parallel" analyst agents: run here as SEQUENTIAL graph edges
rather than true concurrent branches -- a deliberate, documented
simplification for MSc dissertation scope (see earlier design
discussion): the agents ARE conceptually independent, which is the
property that matters architecturally, while Groq's rate limit would
serialise their LLM calls regardless of code structure anyway.

Evaluation-harness scenario pinning: if state.target_period_end /
state.target_macro_date are set (see AgentState docstring), the
Fundamental and Macro Agent nodes pass them through to analyze(),
pinning a specific historical scenario instead of "whatever is most
recent" -- this is what makes the evaluation harness's 5-ticker x
3-time-period design reproducible within a single evaluation session.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from langgraph.graph import END, START, StateGraph

from verifyfin.agents.debate_agent import DebateAgent
from verifyfin.agents.fundamental_agent import FundamentalAgent
from verifyfin.agents.macro_agent import MacroAgent
from verifyfin.agents.sentiment_agent import SentimentAgent
from verifyfin.agents.synthesis_agent import SynthesisAgent
from verifyfin.agents.verification_agent import VerificationAgent
from verifyfin.data_pipeline.vector_store import VectorStore
from verifyfin.evaluation.run_metrics import RunMetrics, collect_run_metrics
from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.state import AgentState, PipelineStage
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VerifyFinPipeline:
    def __init__(
        self,
        config_path: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        cache_namespace: Optional[str] = None,
    ):
        self.config_path = config_path
        self._llm = GroqClient(config_path, cache_namespace=cache_namespace)
        self._store = vector_store if vector_store is not None else VectorStore(config_path)

        self._fundamental_agent = FundamentalAgent(config_path)
        self._fundamental_agent._llm = self._llm
        self._macro_agent = MacroAgent(config_path, llm_client=self._llm)
        self._sentiment_agent = SentimentAgent(config_path, llm_client=self._llm)
        self._verification_agent = VerificationAgent(config_path, vector_store=self._store)
        self._debate_agent = DebateAgent(config_path, llm_client=self._llm)
        self._synthesis_agent = SynthesisAgent(config_path, llm_client=self._llm)

        self._graph = self._build_graph()
        self.last_run_metrics: Optional[RunMetrics] = None

    # ---------- node functions ----------

    def _run_fundamental(self, state: AgentState) -> dict:
        logger.info(f"[Orchestrator] Running Fundamental Agent for {state.ticker}...")
        try:
            claims = self._fundamental_agent.analyze(
                state.ticker, state.time_period, target_period_end=state.target_period_end
            )
        except Exception as e:
            logger.warning(f"Fundamental Agent failed: {e}")
            return {"error_log": state.error_log + [f"Fundamental Agent failed: {e}"]}
        return {"raw_claims": state.raw_claims + claims, "stage": PipelineStage.ANALYSTS_RUNNING}

    def _run_macro(self, state: AgentState) -> dict:
        logger.info(f"[Orchestrator] Running Macro Agent for {state.ticker}...")
        try:
            claims = self._macro_agent.analyze(
                state.ticker, state.time_period, target_date=state.target_macro_date
            )
        except Exception as e:
            logger.warning(f"Macro Agent failed: {e}")
            return {"error_log": state.error_log + [f"Macro Agent failed: {e}"]}
        return {"raw_claims": state.raw_claims + claims}

    def _run_sentiment(self, state: AgentState) -> dict:
        logger.info(f"[Orchestrator] Running Sentiment Agent for {state.ticker}...")
        try:
            claims = self._sentiment_agent.analyze(state.ticker, state.time_period)
        except Exception as e:
            logger.warning(f"Sentiment Agent failed: {e}")
            return {"error_log": state.error_log + [f"Sentiment Agent failed: {e}"]}
        return {"raw_claims": state.raw_claims + claims, "stage": PipelineStage.ANALYSTS_COMPLETE}

    def _run_verification(self, state: AgentState) -> dict:
        logger.info(f"[Orchestrator] Verifying {len(state.raw_claims)} claims...")
        verified = self._verification_agent.verify_claims(list(state.raw_claims))
        return {"verified_claims": verified, "stage": PipelineStage.VERIFICATION_COMPLETE}

    def _run_debate(self, state: AgentState) -> dict:
        logger.info(f"[Orchestrator] Running Debate Agent on CHALLENGED claims...")
        debated = self._debate_agent.debate_claims(list(state.verified_claims))
        return {"debated_claims": debated, "stage": PipelineStage.DEBATE_COMPLETE}

    def _run_synthesis(self, state: AgentState) -> dict:
        logger.info("[Orchestrator] Synthesizing final report...")
        report = self._synthesis_agent.synthesize(state)
        return {"final_report": report, "stage": PipelineStage.COMPLETE}

    # ---------- graph construction ----------

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("fundamental", self._run_fundamental)
        graph.add_node("macro", self._run_macro)
        graph.add_node("sentiment", self._run_sentiment)
        graph.add_node("verification", self._run_verification)
        graph.add_node("debate", self._run_debate)
        graph.add_node("synthesis", self._run_synthesis)

        graph.add_edge(START, "fundamental")
        graph.add_edge("fundamental", "macro")
        graph.add_edge("macro", "sentiment")
        graph.add_edge("sentiment", "verification")
        graph.add_edge("verification", "debate")
        graph.add_edge("debate", "synthesis")
        graph.add_edge("synthesis", END)

        return graph.compile()

    # ---------- public API ----------

    def run(
        self,
        ticker: str,
        time_period: str,
        run_id: Optional[str] = None,
        target_period_end: Optional[str] = None,
        target_macro_date: Optional[str] = None,
    ) -> AgentState:
        """
        Runs the full pipeline for one ticker/time-period and returns
        the final AgentState.

        target_period_end / target_macro_date: if given, pin the
        Fundamental/Macro Agents to a specific historical scenario
        (see AgentState docstring) instead of "whatever is most
        recent" -- used by the evaluation harness for reproducible
        multi-period comparisons.

        Also collects a RunMetrics record, stored on
        self.last_run_metrics.
        """
        run_id = run_id or str(uuid.uuid4())
        initial_state = AgentState(
            run_id=run_id,
            ticker=ticker.upper(),
            time_period=time_period,
            target_period_end=target_period_end,
            target_macro_date=target_macro_date,
            stage=PipelineStage.INITIALISED,
        )

        self._llm.reset_stats()
        self._store.reset_stats()

        logger.info(f"[Orchestrator] Starting pipeline run {initial_state.run_id} for {ticker} ({time_period})")
        start = time.perf_counter()
        result = self._graph.invoke(initial_state)
        runtime_seconds = time.perf_counter() - start

        final_state = result if isinstance(result, AgentState) else AgentState(**result)

        claims_for_metrics = final_state.debated_claims if final_state.debated_claims else final_state.verified_claims
        self.last_run_metrics = collect_run_metrics(
            run_id=final_state.run_id,
            ticker=ticker.upper(),
            time_period=time_period,
            condition="verify_fin",
            runtime_seconds=runtime_seconds,
            claims=claims_for_metrics,
            groq_client=self._llm,
            vector_store=self._store,
        )

        logger.info(f"[Orchestrator] Pipeline run {final_state.run_id} complete in {runtime_seconds:.2f}s.")
        return final_state
