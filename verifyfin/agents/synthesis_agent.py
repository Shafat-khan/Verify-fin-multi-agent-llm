"""
verifyfin.agents.synthesis_agent
===================================
Synthesis Agent (Proposal Objective 2.4): produces the final,
citation-backed research report using ONLY claims that passed
verification (GROUNDED, or CHALLENGED-then-upheld-by-debate) -- see
ClaimObject.is_usable_in_final_report() / AgentState.usable_claims_for_synthesis().

This is the guarantee at the heart of VERIFY-FIN's novelty: a claim
that was BLOCKED, or CHALLENGED-and-rejected, physically cannot reach
this agent's input and therefore cannot appear in the final report --
the LLM here is not asked to "try to be careful"; it structurally
never sees the unverified claims at all. This is the concrete
mechanism behind the project's central hallucination-mitigation claim.
"""

from __future__ import annotations

from typing import Optional

from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.state import AgentState
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial research report writer. You will be given a list of VERIFIED, "
    "source-cited claims about a company. Write a short, coherent research summary using "
    "ONLY these claims.\n"
    "Rules:\n"
    "1. Use ONLY the claims given -- do not add any fact, figure, or opinion not present "
    "in the claims.\n"
    "2. Write in a professional, neutral financial-analyst tone.\n"
    "3. Organise the report with brief natural paragraphs (fundamentals, then macro "
    "context, then sentiment, in that order, skipping any category with no claims).\n"
    "4. Do not invent a title, and do not add a disclaimer -- just the analysis paragraphs.\n"
    "5. Keep it concise: 150-300 words."
)

_NO_CLAIMS_MESSAGE_TEMPLATE = (
    "No claims about {ticker} for {time_period} could be verified against available source "
    "documents. No report can be generated for this period."
)


class SynthesisAgent:
    def __init__(self, config_path: Optional[str] = None, llm_client: Optional[GroqClient] = None):
        self.config_path = config_path
        self._llm = llm_client if llm_client is not None else GroqClient(config_path)

    @staticmethod
    def _format_claims_for_prompt(claims: list) -> str:
        lines = []
        for c in claims:
            category = c.claim_category.value if hasattr(c.claim_category, "value") else c.claim_category
            lines.append(
                f"- [{category}] {c.claim_text} "
                f"(source: {c.source_reference.source_type}, confidence: {c.confidence_score})"
            )
        return "\n".join(lines)

    def synthesize(self, state: AgentState, require_verification: bool = True) -> str:
        """
        Produces the final report text.

        By default (require_verification=True), uses ONLY
        state.usable_claims_for_synthesis() -- i.e. GROUNDED claims, or
        CHALLENGED claims the Debate Agent upheld. This is VERIFY-FIN's
        actual behaviour.

        require_verification=False bypasses that check entirely and
        uses ALL of state.raw_claims regardless of verification status.
        This exists specifically so the Unverified Multi-Agent baseline
        (Proposal §2.5 evaluation design, Condition 2) can reuse this
        exact same agent -- same prompt, same LLM, same formatting --
        with verification as the ONLY variable that differs from the
        full VERIFY-FIN condition. This deliberately reproduces the
        behaviour of prior multi-agent systems (e.g. TradingAgents, per
        the Literature Review) that pass every analyst claim straight
        to synthesis with no grounding check.
        """
        usable_claims = state.raw_claims if not require_verification else state.usable_claims_for_synthesis()

        if not usable_claims:
            logger.warning(f"No usable claims for {state.ticker}; producing a 'no verified claims' report.")
            return _NO_CLAIMS_MESSAGE_TEMPLATE.format(ticker=state.ticker, time_period=state.time_period)

        claims_payload = self._format_claims_for_prompt(usable_claims)
        user_prompt = f"Verified claims about {state.ticker} ({state.time_period}):\n{claims_payload}"

        logger.info(f"Synthesizing report for {state.ticker} from {len(usable_claims)} claims...")
        report = self._llm.complete(user_prompt=user_prompt, system_prompt=_SYSTEM_PROMPT)
        return report.strip()
