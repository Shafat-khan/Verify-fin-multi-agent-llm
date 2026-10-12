"""
verifyfin.schemas.state
=========================

Defines the AgentState: the single typed, Pydantic-validated object that
flows through every node of the LangGraph StateGraph pipeline.

Rationale (Literature Review, 2.3.5):
    VERIFY-FIN uses LangGraph rather than AutoGen specifically because
    LangGraph's StateGraph enforces a typed Pydantic state object that
    passes between all agents, ensuring every transition in the pipeline
    is schema-validated -- unlike AutoGen's unstructured conversational
    strings, which require brittle downstream parsing.

Evaluation-harness fields (target_period_end, target_macro_date): added
so the evaluation harness can pin a SPECIFIC historical scenario
(a specific quarter's fundamentals, a specific date's macro snapshot)
through the whole LangGraph pipeline. Node functions only receive
`state`, so this is the only clean way to pass scenario-specific
parameters into agent calls without threading extra arguments through
LangGraph's node-function signature (which the framework doesn't
support). Both default to None, meaning "use the current/most recent
data" -- the normal behaviour for a one-off analysis run outside the
evaluation harness.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from verifyfin.schemas.claim import ClaimObject


class PipelineStage(str, Enum):
    """Tracks where in the pipeline the state currently is. Used by the Orchestrator for routing."""

    INITIALISED = "initialised"
    ANALYSTS_RUNNING = "analysts_running"
    ANALYSTS_COMPLETE = "analysts_complete"
    VERIFICATION_RUNNING = "verification_running"
    VERIFICATION_COMPLETE = "verification_complete"
    DEBATE_RUNNING = "debate_running"
    DEBATE_COMPLETE = "debate_complete"
    SYNTHESIS_RUNNING = "synthesis_running"
    COMPLETE = "complete"
    FAILED = "failed"


class AgentState(BaseModel):
    """
    The complete state passed between all LangGraph nodes for a single
    ticker/time-period analysis run.

    Design note: this state object is intentionally the ONLY channel of
    communication between agents. No agent should communicate via
    free-form strings outside this object.
    """

    run_id: str
    ticker: str
    time_period: str
    condition: str = Field(
        default="verify_fin",
        description="Which experimental condition this run belongs to: 'single_agent', 'unverified_multiagent', or 'verify_fin'.",
    )

    # Evaluation-harness scenario pinning (see module docstring).
    target_period_end: Optional[str] = Field(
        default=None, description="Pinned fundamental-data quarter (period-end date, 'YYYY-MM-DD'), if set."
    )
    target_macro_date: Optional[str] = Field(
        default=None, description="Pinned macro-data target date ('YYYY-MM-DD'), if set."
    )

    stage: PipelineStage = Field(default=PipelineStage.INITIALISED)

    raw_claims: list[ClaimObject] = Field(default_factory=list)
    verified_claims: list[ClaimObject] = Field(default_factory=list)
    debated_claims: list[ClaimObject] = Field(default_factory=list)

    final_report: Optional[str] = Field(default=None)

    conflicts_detected: list[str] = Field(
        default_factory=list, description="Human-readable log of conflicting claims the Orchestrator had to resolve."
    )
    error_log: list[str] = Field(default_factory=list)

    def add_raw_claim(self, claim: ClaimObject) -> None:
        self.raw_claims.append(claim)

    def usable_claims_for_synthesis(self) -> list[ClaimObject]:
        """Claims the Synthesis Agent is permitted to use in the final report."""
        pool = self.debated_claims if self.debated_claims else self.verified_claims
        return [c for c in pool if c.is_usable_in_final_report()]

    def hallucination_rate(self) -> float:
        """
        Primary evaluation metric (Proposal §2.5): proportion of claims
        that could NOT be grounded in source documents.
        """
        if not self.verified_claims:
            return 0.0
        blocked = sum(1 for c in self.verified_claims if not c.is_usable_in_final_report())
        return blocked / len(self.verified_claims)

    model_config = ConfigDict(arbitrary_types_allowed=True)
