"""
verifyfin.evaluation.posthoc_factcheck
=========================================
Applies VERIFY-FIN's own Verification + Debate pipeline, as a
MEASUREMENT-ONLY pass, to the two baseline conditions -- enabling a
genuine, fair, 3-way hallucination-rate comparison (Proposal §2.5).

This never changes what a condition actually produced (its own
report/claims stay exactly as they were). It only measures, after the
fact, what fraction of its claims the SAME verification mechanism
VERIFY-FIN uses would flag -- using the SAME scenario's knowledge base,
so the comparison is apples-to-apples.

Both Verification AND Debate are run here, not just Verification --
otherwise a claim that would have been CHALLENGED-then-upheld in
VERIFY-FIN's own pipeline would count as "not usable" in a baseline's
post-hoc measurement, unfairly inflating its apparent hallucination
rate relative to VERIFY-FIN for no real reason.

Two entry points, matching the two conditions that need this:
    - factcheck_unverified_baseline_claims(): the Unverified
      Multi-Agent baseline's raw_claims already have real, structured
      SourceReferences (from the same real analyst agents VERIFY-FIN
      uses) -- they just skip verification/debate by design. Run them
      through both stages directly.
    - factcheck_single_agent_report(): the Single-Agent baseline
      produces free prose with no claim structure at all. A dedicated
      extraction step first asks an LLM to pull out discrete, checkable
      factual claims from the text, each wrapped in a minimal
      ClaimObject (with NO source reference of its own, since the
      baseline never cited anything) before being run through the same
      Verification + Debate pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from verifyfin.agents.debate_agent import DebateAgent
from verifyfin.agents.verification_agent import VerificationAgent
from verifyfin.data_pipeline.vector_store import VectorStore
from verifyfin.evaluation.run_metrics import RunMetrics, apply_posthoc_verification
from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, SourceReference
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "You are analysing a financial research report for fact-checking purposes. Extract "
    "every discrete, checkable factual claim from the text below -- each claim should be a "
    "single, standalone factual statement (e.g. a specific number, date, percentage, or "
    "concrete assertion about performance, the economy, or sentiment). Do not include "
    "vague or purely stylistic sentences with no checkable factual content.\n"
    "Reply with ONLY a JSON array of strings, no other text, in this exact format:\n"
    '["claim 1", "claim 2", ...]'
)


def extract_claims_from_text(llm_client: GroqClient, report_text: str) -> list[str]:
    """
    Uses an LLM call to decompose free-form report text into discrete,
    checkable factual claim strings. Returns an empty list (not an
    error) if the text has no content or extraction genuinely fails --
    callers should treat an empty result as "nothing could be
    fact-checked", which is itself meaningful information, not a bug.

    max_tokens=3000 (raised from an earlier, still-too-low 2000): a
    report with 20+ extractable claims needs real room for its JSON
    array response. Note: an earlier attempt to fix this by raising
    max_tokens alone did NOT work, because GroqClient's cache key did
    not include max_tokens -- a stale, truncated response cached under
    the OLD token budget kept being served regardless of this fix. That
    is now fixed at the GroqClient level (max_tokens is part of the
    cache key), so this parameter change will actually take effect.

    JSON-repair fallback: even with a generous token budget, a
    genuinely long report can still produce a response cut off mid-way
    through the JSON array (e.g. "Unterminated string starting at...").
    Rather than discarding the ENTIRE response when strict json.loads()
    fails, _repair_truncated_claim_array() recovers every syntactically
    COMPLETE string literal that was actually returned before the cutoff
    -- so a 20-claim response truncated after 18 claims still yields 18
    real, usable claims instead of zero.
    """
    if not report_text or not report_text.strip():
        return []

    raw_response = llm_client.complete(
        user_prompt=report_text, system_prompt=_CLAIM_EXTRACTION_SYSTEM_PROMPT, max_tokens=3000
    )

    text = raw_response.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Claim extraction response was not valid JSON ({e}); attempting repair.")
        repaired = _repair_truncated_claim_array(text)
        if repaired:
            logger.warning(f"Repair recovered {len(repaired)} complete claim(s) from the truncated response.")
            return repaired
        logger.warning("Repair recovered no complete claims; treating as no claims extracted.")
        return []

    if not isinstance(parsed, list):
        return []
    return [c for c in parsed if isinstance(c, str) and c.strip()]


def _repair_truncated_claim_array(text: str) -> list[str]:
    """
    Recovers complete, well-formed JSON string literals from a response
    that failed strict json.loads() -- typically because the array was
    cut off mid-string when the response hit its token limit. Finds
    every substring matching a complete JSON string literal pattern
    (opening quote ... closing quote, with escaped characters handled)
    and decodes each one, silently skipping only the final, genuinely
    incomplete entry (the one actually cut off) rather than losing the
    entire array. Returns [] if nothing matches, which callers already
    treat as "no claims extracted", not an error.
    """
    matches = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
    recovered = []
    for m in matches:
        try:
            recovered.append(json.loads(f'"{m}"'))
        except json.JSONDecodeError:
            continue
    # The system prompt/user-prompt-echo can occasionally match this
    # pattern too if the model repeats instruction text; a claim string
    # extracted from a genuine claims array is never empty and is
    # always more than a couple of characters, so trivially short
    # matches are dropped as noise rather than counted as claims.
    return [c for c in recovered if len(c.strip()) > 5]


def _build_claim_objects(claim_texts: list[str], ticker: str, time_period: str) -> list[ClaimObject]:
    """
    Wraps extracted claim strings in minimal ClaimObjects for the
    Verification Agent. These have NO source reference of their own
    (excerpt=None) since the Single-Agent baseline never cited
    anything -- VerificationAgent already handles a missing excerpt
    gracefully (falls back to semantic-only matching, no numeric
    shortcut available), which is exactly the fair, honest treatment
    an uncited claim deserves.
    """
    return [
        ClaimObject(
            claim_text=text,
            claim_category=ClaimCategory.NUMERICAL,  # coarse default -- extraction doesn't distinguish category
            produced_by=AgentRole.FUNDAMENTAL,  # placeholder: this claim wasn't "produced by" any real analyst agent
            source_reference=SourceReference(source_type="single_agent_extraction", document_id="n/a", excerpt=None),
            confidence_score=0.5,  # arbitrary -- the Single-Agent baseline never reported a real confidence
            ticker=ticker.upper(),
            time_period=time_period,
        )
        for text in claim_texts
    ]


def _run_verification_and_debate(
    claims: list[ClaimObject], vector_store: VectorStore, llm_client: GroqClient, config_path: Optional[str] = None
) -> list[ClaimObject]:
    verification_agent = VerificationAgent(config_path, vector_store=vector_store)
    debate_agent = DebateAgent(config_path, llm_client=llm_client)

    verified = verification_agent.verify_claims(list(claims))
    debated = debate_agent.debate_claims(verified)
    return debated


def factcheck_unverified_baseline_claims(
    raw_claims: list[ClaimObject],
    vector_store: VectorStore,
    llm_client: GroqClient,
    metrics: RunMetrics,
    config_path: Optional[str] = None,
) -> list[ClaimObject]:
    """
    Runs the Unverified Multi-Agent baseline's raw_claims through the
    same Verification + Debate pipeline VERIFY-FIN uses, for
    measurement only, and populates metrics.posthoc_* fields in place.
    Returns the fact-checked claims (empty list if there were none), so
    callers can persist them for inspection or human labelling.
    """
    if not raw_claims:
        metrics.posthoc_total_claims = 0
        return []

    fact_checked = _run_verification_and_debate(raw_claims, vector_store, llm_client, config_path)
    apply_posthoc_verification(metrics, fact_checked)
    return fact_checked


def factcheck_single_agent_report(
    report_text: str,
    ticker: str,
    time_period: str,
    vector_store: VectorStore,
    llm_client: GroqClient,
    metrics: RunMetrics,
    config_path: Optional[str] = None,
) -> list[ClaimObject]:
    """
    Extracts discrete claims from a free-text baseline report (Single-Agent
    or Single-Agent + RAG), then runs them through the same Verification +
    Debate pipeline, populating metrics.posthoc_* fields in place.
    Returns the fact-checked claims ([] if none could be extracted), so
    callers can persist them for inspection or human labelling.
    """
    claim_texts = extract_claims_from_text(llm_client, report_text)

    if not claim_texts:
        metrics.posthoc_total_claims = 0
        metrics.posthoc_extraction_failed = True
        logger.warning(f"No claims could be extracted from the Single-Agent report for {ticker} ({time_period}).")
        return []

    claims = _build_claim_objects(claim_texts, ticker, time_period)
    fact_checked = _run_verification_and_debate(claims, vector_store, llm_client, config_path)
    apply_posthoc_verification(metrics, fact_checked)
    return fact_checked
