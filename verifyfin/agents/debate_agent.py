"""
verifyfin.agents.debate_agent
================================
Debate Agent (Proposal Objective 2.4; Literature Review 2.5.4/Gap 5):
a second-opinion LLM entailment check, invoked ONLY for claims the
Verification Agent routed to CHALLENGED. GROUNDED and BLOCKED claims
never reach this agent -- GROUNDED claims are already confidently
supported, and BLOCKED claims already failed decisively.

Why this exists: cosine similarity is a single, imperfect signal (see
verification_agent module docstring for real examples of why a
correct claim can still score in the borderline band -- e.g. a table
row containing multiple comparison-period numbers is a "noisier"
semantic match than a short two-column row, even when the correct
number was found). The Debate Agent asks a narrower, more direct
question than embedding similarity can: "does this specific evidence
text support this specific claim?" -- a lightweight entailment check,
in the same spirit as (but much cheaper than) RAGAS Faithfulness,
which VERIFY-FIN also uses as its secondary evaluation metric.

Fail-safe design: if the LLM's response can't be parsed, or if a
CHALLENGED claim has no retrieved evidence at all, the claim is
REJECTED, not upheld. For a hallucination-mitigation system, defaulting
to exclusion under uncertainty is the safer failure mode than silently
including a claim that couldn't be confidently verified.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.claim import ClaimObject, GroundingVerdict
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a rigorous fact-checking assistant. You will be given a CLAIM and a piece "
    "of EVIDENCE retrieved from a company's official source document. Your job is to "
    "determine whether the evidence directly supports the claim.\n"
    "Rules:\n"
    "- 'upheld' means the evidence contains the same fact/number as the claim, referring "
    "to the same concept and time period.\n"
    "- 'rejected' means the evidence does NOT clearly support the claim -- e.g. a "
    "different number, a different time period, an unrelated concept, or evidence that "
    "is ambiguous or inconclusive.\n"
    "- Be strict: if you are not confident, respond 'rejected'.\n"
    "Reply with ONLY a JSON object, no other text, in this exact format:\n"
    '{"verdict": "upheld" or "rejected", "reasoning": "<one short sentence>"}'
)

_NO_EVIDENCE_PLACEHOLDER = "(no evidence retrieved from knowledge base)"


class DebateAgent:
    def __init__(self, config_path: Optional[str] = None, llm_client: Optional[GroqClient] = None):
        self.config_path = config_path
        # Allow injecting a shared GroqClient (reuses the same disk
        # cache and rate limiter as other agents in the pipeline)
        # rather than always constructing a fresh one.
        self._llm = llm_client if llm_client is not None else GroqClient(config_path)

    @staticmethod
    def _parse_debate_response(raw_response: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parses the LLM's JSON verdict response. Tolerant of markdown
        code fences. Returns (verdict, reasoning), or (None, None) if
        the response can't be parsed -- the caller treats this as a
        fail-closed "rejected".
        """
        text = raw_response.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Debate Agent response as JSON ({e}). Response was: {raw_response[:200]}")
            return None, None

        verdict = parsed.get("verdict") if isinstance(parsed, dict) else None
        reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None

        if verdict not in ("upheld", "rejected"):
            logger.warning(f"Debate Agent returned an unrecognised verdict: {verdict!r}")
            return None, reasoning

        return verdict, reasoning

    def debate_claim(self, claim: ClaimObject) -> ClaimObject:
        """
        Runs the entailment check for a single CHALLENGED claim IN PLACE
        (mutates and returns it), setting claim.debate_resolution to
        "upheld" or "rejected".

        Claims that are not CHALLENGED are returned unchanged -- this
        method is safe to call on any claim, but only does work for
        CHALLENGED ones, matching the architecture's "debate only
        borderline claims" design.
        """
        if claim.verdict != GroundingVerdict.CHALLENGED:
            return claim

        if not claim.retrieved_evidence or claim.retrieved_evidence == _NO_EVIDENCE_PLACEHOLDER:
            logger.info(f"No evidence to debate for claim '{claim.claim_text[:60]}...'; rejecting.")
            claim.debate_resolution = "rejected"
            return claim

        user_prompt = f"CLAIM: {claim.claim_text}\n\nEVIDENCE: {claim.retrieved_evidence}"
        raw_response = self._llm.complete(user_prompt=user_prompt, system_prompt=_SYSTEM_PROMPT)
        verdict, reasoning = self._parse_debate_response(raw_response)

        if verdict is None:
            # Fail-closed: an unparseable or unrecognised response must
            # not silently let a claim through.
            claim.debate_resolution = "rejected"
        else:
            claim.debate_resolution = verdict

        logger.info(
            f"Debate resolved '{claim.debate_resolution}' for claim '{claim.claim_text[:60]}...' "
            f"(reasoning: {reasoning})"
        )
        return claim

    def debate_claims(self, claims: list[ClaimObject]) -> list[ClaimObject]:
        """
        Runs debate on every CHALLENGED claim in a batch; claims with
        other verdicts pass through untouched, with no LLM call spent
        on them. Returns the same list, mutated in place.
        """
        challenged = [c for c in claims if c.verdict == GroundingVerdict.CHALLENGED]
        logger.info(f"Debate Agent reviewing {len(challenged)} CHALLENGED claim(s) out of {len(claims)} total.")

        for claim in challenged:
            self.debate_claim(claim)

        upheld = sum(1 for c in challenged if c.debate_resolution == "upheld")
        rejected = sum(1 for c in challenged if c.debate_resolution == "rejected")
        logger.info(f"Debate complete: {upheld} upheld, {rejected} rejected.")

        return claims
