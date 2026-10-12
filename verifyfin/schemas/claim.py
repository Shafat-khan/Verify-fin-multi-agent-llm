"""
verifyfin.schemas.claim
========================

Defines the ClaimObject schema: the structured, Pydantic-validated
inter-agent communication format that is central to VERIFY-FIN's novelty.

Rationale (Literature Review, Gap 2):
    Existing multi-agent financial systems (TradingAgents, AutoGen-based
    pipelines) pass free-form prose between agents. This makes it
    impossible to enforce source citation or confidence scoring, and
    requires brittle downstream parsing. VERIFY-FIN instead mandates that
    every analyst agent emits a ClaimObject: it is structurally impossible
    for an agent to produce a claim without an explicit source_reference.

Rationale (Literature Review, 2.6.3):
    Different claim types require different verification strategies
    (e.g. a numerical claim can be grounded via cosine similarity against
    a retrieved passage, but a sentiment claim like "market sentiment is
    positive" cannot be verified the same way). The `claim_category`
    field lets the Verification Agent branch its logic accordingly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClaimCategory(str, Enum):
    """Type of factual assertion. Drives verification strategy downstream."""

    NUMERICAL = "numerical"        # e.g. "AAPL reported revenue of $94.9B in Q3 2024"
    SENTIMENT = "sentiment"        # e.g. "Market sentiment towards AAPL is strongly positive"
    MACRO = "macro"                # e.g. "The Federal Funds Rate is 5.25%-5.50%"
    QUALITATIVE = "qualitative"    # e.g. "The company faces increased competitive pressure"


class AgentRole(str, Enum):
    """Which analyst agent produced this claim. Used for provenance/traceability."""

    FUNDAMENTAL = "fundamental_agent"
    SENTIMENT = "sentiment_agent"
    MACRO = "macro_agent"


class GroundingVerdict(str, Enum):
    """
    Continuous grounding score -> 3-way routing decision made by the
    Verification Agent (Literature Review, Gap 3 — contrasts with FEVER's
    discrete SUPPORTED/REFUTED/NEI verdicts).
    """

    UNVERIFIED = "unverified"      # not yet passed through the Verification Agent
    GROUNDED = "grounded"          # grounding_score >= grounded_threshold
    CHALLENGED = "challenged"      # between challenged_threshold and grounded_threshold -> Debate Agent
    BLOCKED = "blocked"            # grounding_score < challenged_threshold -> excluded from final report


class SourceReference(BaseModel):
    """
    Explicit citation for a claim. Every ClaimObject must carry one,
    per Gap 2. This is what makes the final VERIFY-FIN report
    traceable and citation-backed (Joshi, 2025 — explainability gap).
    """

    source_type: str = Field(
        ...,
        description="Origin of the source, e.g. 'edgar_10k', 'edgar_10q', 'yfinance', 'fred', 'news'.",
    )
    document_id: str = Field(
        ..., description="Identifier of the source document/series, e.g. accession number, ticker, FRED series id."
    )
    excerpt: Optional[str] = Field(
        default=None,
        description="Optional short excerpt from the source the agent is citing, used by the Verification Agent as the retrieval query if provided.",
    )
    retrieved_at: Optional[datetime] = Field(default=None)


class ClaimObject(BaseModel):
    """
    The core structured unit passed between all VERIFY-FIN agents.

    An analyst agent MUST populate claim_text, source_reference,
    confidence_score, and claim_category. It is invalid (raises a
    pydantic ValidationError) to construct a ClaimObject without a
    source reference -- this is the schema-level enforcement described
    in the Literature Review as VERIFY-FIN's key architectural
    differentiator from prose-based multi-agent systems.
    """

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str = Field(..., min_length=1, description="The factual assertion in natural language.")
    claim_category: ClaimCategory
    produced_by: AgentRole
    source_reference: SourceReference
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Agent's self-reported confidence in this claim (Self-RAG-style reflection, Lit Review 2.5.4)."
    )
    ticker: str = Field(..., description="Stock ticker this claim pertains to.")
    time_period: str = Field(..., description="Evaluation time period this claim pertains to, e.g. '2024-Q1'.")

    # --- Populated later, by the Verification Agent ---
    grounding_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Cosine similarity between claim and retrieved evidence."
    )
    verdict: GroundingVerdict = Field(default=GroundingVerdict.UNVERIFIED)
    retrieved_evidence: Optional[str] = Field(
        default=None, description="The evidence chunk retrieved by the Verification Agent to score this claim."
    )

    # --- Populated later, only if verdict == CHALLENGED ---
    debate_resolution: Optional[str] = Field(
        default=None, description="Outcome of the Debate Agent's second-opinion entailment check, if invoked."
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("claim_text")
    @classmethod
    def claim_text_must_not_be_placeholder(cls, v: str) -> str:
        if v.strip().lower() in {"", "n/a", "todo", "tbd"}:
            raise ValueError("claim_text cannot be empty or a placeholder value.")
        return v.strip()

    def mark_grounded(self, grounding_score: float, evidence: str, threshold_grounded: float, threshold_challenged: float) -> None:
        """Apply the Verification Agent's 3-way routing logic to this claim in place."""
        self.grounding_score = grounding_score
        self.retrieved_evidence = evidence
        if grounding_score >= threshold_grounded:
            self.verdict = GroundingVerdict.GROUNDED
        elif grounding_score >= threshold_challenged:
            self.verdict = GroundingVerdict.CHALLENGED
        else:
            self.verdict = GroundingVerdict.BLOCKED

    def is_usable_in_final_report(self) -> bool:
        """Only GROUNDED claims, or CHALLENGED claims resolved favourably by the Debate Agent, reach Synthesis."""
        if self.verdict == GroundingVerdict.GROUNDED:
            return True
        if self.verdict == GroundingVerdict.CHALLENGED and self.debate_resolution == "upheld":
            return True
        return False

    model_config = ConfigDict(
        use_enum_values=False,
        json_schema_extra={
            "example": {
                "claim_text": "AAPL reported revenue of $94.9B in Q3 2024.",
                "claim_category": "numerical",
                "produced_by": "fundamental_agent",
                "source_reference": {
                    "source_type": "edgar_10q",
                    "document_id": "0000320193-24-000081",
                    "excerpt": "Total net sales of $94.9 billion",
                },
                "confidence_score": 0.92,
                "ticker": "AAPL",
                "time_period": "2024-Q3",
            }
        },
    )
