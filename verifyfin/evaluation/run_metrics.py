"""
verifyfin.evaluation.run_metrics
===================================
RunMetrics: a structured record of everything measurable about a
single pipeline/baseline run, collected automatically for the
evaluation harness's comparison graphs.

Post-hoc fact-check fields (added for a genuine 3-way hallucination-
rate comparison -- Proposal §2.5's three-condition evaluation design):
VERIFY-FIN's own grounded/challenged/blocked counts reflect real,
pipeline-gating verification. The Unverified Multi-Agent baseline's
claims were never checked at all (that's the condition's defining
property), and the Single-Agent baseline produces free text with no
claim structure whatsoever -- neither has a verification-derived
number to compare against VERIFY-FIN's out of the box. The posthoc_*
fields hold a SEPARATE, measurement-only verification pass applied
uniformly after the fact: the same VerificationAgent, against the same
scenario's knowledge base, run on (a) the baseline's raw claims
directly (Unverified Multi-Agent) or (b) claims extracted from its
free text via a dedicated extraction step (Single-Agent). This never
changes what the condition actually produced (its own report/claims
are untouched) -- it only measures, after the fact, what fraction of
its claims a fair, uniform check would flag.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from verifyfin.schemas.claim import GroundingVerdict


class RunMetrics(BaseModel):
    run_id: str
    ticker: str
    time_period: str
    condition: str  # "single_agent" | "unverified_multiagent" | "verify_fin"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    total_claims: int = 0
    grounded_count: int = 0
    challenged_count: int = 0
    blocked_count: int = 0
    debate_upheld_count: int = 0
    debate_rejected_count: int = 0
    usable_claims_count: int = 0

    runtime_seconds: float = 0.0
    retrieval_query_count: int = 0
    retrieval_total_latency_seconds: float = 0.0

    llm_call_count: int = 0
    llm_cache_hit_count: int = 0
    llm_api_call_count: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None

    # Post-hoc fact-check fields (see module docstring). None when not
    # applicable (VERIFY-FIN doesn't need a separate pass -- its own
    # fields above already are the fair, uniform measurement).
    posthoc_total_claims: Optional[int] = None
    posthoc_grounded_count: Optional[int] = None
    posthoc_challenged_count: Optional[int] = None
    posthoc_blocked_count: Optional[int] = None
    posthoc_usable_count: Optional[int] = None
    posthoc_extraction_failed: bool = False  # True if claim extraction (single_agent only) produced nothing usable

    def hallucination_rate(self) -> Optional[float]:
        """
        This condition's OWN gating hallucination rate (meaningful for
        VERIFY-FIN; will read as 100% for unverified_multiagent, which
        means "nothing was checked", not "everything was wrong" -- use
        posthoc_hallucination_rate() for a fair cross-condition number).
        """
        if self.total_claims == 0:
            return None
        return 1.0 - (self.usable_claims_count / self.total_claims)

    def posthoc_hallucination_rate(self) -> Optional[float]:
        """
        The fair, uniform, cross-condition-comparable hallucination
        rate: proportion of this condition's claims (or extracted
        claims, for single_agent) that a uniform post-hoc verification
        pass would flag as not usable.
        """
        if not self.posthoc_total_claims:
            return None
        return 1.0 - (self.posthoc_usable_count / self.posthoc_total_claims)


def collect_run_metrics(
    run_id: str,
    ticker: str,
    time_period: str,
    condition: str,
    runtime_seconds: float,
    claims: Optional[list] = None,
    groq_client=None,
    vector_store=None,
) -> RunMetrics:
    metrics = RunMetrics(
        run_id=run_id,
        ticker=ticker,
        time_period=time_period,
        condition=condition,
        runtime_seconds=round(runtime_seconds, 4),
    )

    if claims:
        metrics.total_claims = len(claims)
        metrics.grounded_count = sum(1 for c in claims if c.verdict == GroundingVerdict.GROUNDED)
        metrics.challenged_count = sum(1 for c in claims if c.verdict == GroundingVerdict.CHALLENGED)
        metrics.blocked_count = sum(1 for c in claims if c.verdict == GroundingVerdict.BLOCKED)
        metrics.debate_upheld_count = sum(1 for c in claims if c.debate_resolution == "upheld")
        metrics.debate_rejected_count = sum(1 for c in claims if c.debate_resolution == "rejected")
        metrics.usable_claims_count = sum(1 for c in claims if c.is_usable_in_final_report())

    if groq_client is not None:
        llm_stats = groq_client.get_stats()
        metrics.llm_call_count = llm_stats["call_count"]
        metrics.llm_cache_hit_count = llm_stats["cache_hit_count"]
        metrics.llm_api_call_count = llm_stats["api_call_count"]
        metrics.llm_prompt_tokens = llm_stats["total_prompt_tokens"]
        metrics.llm_completion_tokens = llm_stats["total_completion_tokens"]
        metrics.llm_total_tokens = llm_stats["total_tokens"]

    if vector_store is not None:
        retrieval_stats = vector_store.get_stats()
        metrics.retrieval_query_count = retrieval_stats["query_count"]
        metrics.retrieval_total_latency_seconds = retrieval_stats["total_query_time_seconds"]

    return metrics


def apply_posthoc_verification(metrics: RunMetrics, verified_claims: list) -> None:
    """
    Populates metrics.posthoc_* fields from a list of claims that have
    already been run through VerificationAgent.verify_claims() for
    measurement purposes. Mutates metrics in place.
    """
    metrics.posthoc_total_claims = len(verified_claims)
    metrics.posthoc_grounded_count = sum(1 for c in verified_claims if c.verdict == GroundingVerdict.GROUNDED)
    metrics.posthoc_challenged_count = sum(1 for c in verified_claims if c.verdict == GroundingVerdict.CHALLENGED)
    metrics.posthoc_blocked_count = sum(1 for c in verified_claims if c.verdict == GroundingVerdict.BLOCKED)
    metrics.posthoc_usable_count = sum(1 for c in verified_claims if c.is_usable_in_final_report())
