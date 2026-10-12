"""
verifyfin.agents.fundamental_agent
=====================================
Fundamental Agent (Proposal Objective 2.3): produces structured,
source-cited ClaimObjects about a ticker's financial fundamentals,
sourced entirely from SEC EDGAR's official XBRL Company Facts API.

Key design decision -- LLM phrases, code sources: the LLM is
deliberately used ONLY to phrase already-retrieved, already-verified
numbers into natural language. Every numeric value and every
SourceReference comes directly from EdgarXbrlClient's structured,
official output -- deterministic and Python-side, not LLM-generated.

Multi-period support (added for the evaluation harness's "5 tickers x
3 time periods" design): analyze() accepts an optional
target_period_end, letting the evaluation harness pin a SPECIFIC
historical quarter (resolved once via
EdgarXbrlClient.get_available_quarterly_periods()) rather than always
getting whatever is currently most recent -- the latter silently
drifts as new filings are published, which caused a real
reproducibility bug earlier in development.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from verifyfin.data_pipeline.edgar_xbrl_client import EdgarXbrlClient
from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, SourceReference
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_CONCEPT_LABELS = {
    "revenue": "total revenue",
    "net_income": "net income",
    "eps_basic": "basic earnings per share",
    "eps_diluted": "diluted earnings per share",
    "total_assets": "total assets",
    "total_liabilities": "total liabilities",
    "stockholders_equity": "stockholders' equity",
    "operating_income": "operating income",
}

_SYSTEM_PROMPT = (
    "You are a precise financial analyst assistant. You will be given a list of "
    "financial figures already extracted from a company's official SEC filing. "
    "Your ONLY job is to phrase each figure as a single, clear, factual sentence "
    "for an investment research report.\n"
    "Rules:\n"
    "1. Use ONLY the numbers given to you -- never invent, estimate, or adjust any figure.\n"
    "2. Do not add any information not present in the data given.\n"
    "3. Do not use words like 'approximately' and do not round the numbers.\n"
    "4. Reply with ONLY a JSON array, no other text, in this exact format:\n"
    '[{"concept": "<concept_id>", "claim_text": "<sentence>"}, ...]\n'
    "5. Include exactly one entry per concept you were given."
)


class FundamentalAgent:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._xbrl = EdgarXbrlClient(config_path)
        self._llm = GroqClient(config_path)

    # ---------- LLM phrasing step ----------

    def _phrase_concepts_as_claims(self, ticker: str, concept_data: dict[str, dict]) -> dict[str, str]:
        payload_for_llm = [
            {
                "concept": concept_id,
                "label": _CONCEPT_LABELS[concept_id],
                "value": entry["value"],
                "unit": entry["unit"],
                "period_end": entry["period_end"],
                "form": entry["form"],
            }
            for concept_id, entry in concept_data.items()
        ]

        user_prompt = (
            f"Company ticker: {ticker}\n\n"
            f"Figures (from the filing indicated):\n{json.dumps(payload_for_llm, indent=2)}"
        )

        raw_response = self._llm.complete(user_prompt=user_prompt, system_prompt=_SYSTEM_PROMPT)
        phrased = self._parse_llm_claims(raw_response)

        result = {}
        for concept_id, entry in concept_data.items():
            if concept_id in phrased:
                result[concept_id] = phrased[concept_id]
            else:
                logger.warning(f"LLM did not return a phrased claim for '{concept_id}'; using deterministic fallback.")
                result[concept_id] = (
                    f"{ticker}'s {_CONCEPT_LABELS[concept_id]} was {entry['value']} {entry['unit']} "
                    f"as of {entry['period_end']} (per {entry['form']} filing)."
                )
        return result

    @staticmethod
    def _parse_llm_claims(raw_response: str) -> dict[str, str]:
        text = raw_response.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse LLM claims JSON ({e}); falling back to deterministic phrasing for all concepts."
            )
            return {}

        result = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "concept" in item and "claim_text" in item:
                    result[item["concept"]] = item["claim_text"]
        return result

    # ---------- claim construction ----------

    def _build_fundamental_claims(
        self, ticker: str, time_period: str, target_period_end: Optional[str] = None
    ) -> list[ClaimObject]:
        if target_period_end is not None:
            fundamentals = self._xbrl.get_fundamentals_for_period(ticker, target_period_end)
        else:
            fundamentals = self._xbrl.get_latest_fundamentals(ticker)

        concept_data = {
            concept: entry
            for concept, entry in fundamentals.items()
            if concept not in ("ticker", "period_end") and entry is not None
        }

        if not concept_data:
            logger.warning(f"No XBRL fundamentals available for {ticker}; no fundamental claims produced.")
            return []

        phrased_claims = self._phrase_concepts_as_claims(ticker, concept_data)

        claims = []
        for concept_id, entry in concept_data.items():
            claims.append(
                ClaimObject(
                    claim_text=phrased_claims[concept_id],
                    claim_category=ClaimCategory.NUMERICAL,
                    produced_by=AgentRole.FUNDAMENTAL,
                    source_reference=SourceReference(
                        source_type="edgar_xbrl",
                        document_id=entry["accession_number"],
                        excerpt=(
                            f"{entry['xbrl_tag']}: {entry['value']} {entry['unit']} "
                            f"(period {entry.get('period_start', 'N/A')} to {entry['period_end']})"
                        ),
                    ),
                    confidence_score=0.95,
                    ticker=ticker.upper(),
                    time_period=time_period,
                )
            )
        return claims

    # ---------- public API ----------

    def analyze(self, ticker: str, time_period: str, target_period_end: Optional[str] = None) -> list[ClaimObject]:
        """
        Produces all Fundamental Agent claims for a ticker, sourced from
        XBRL financial statement data.

        If target_period_end is given (a period-end date string, e.g.
        from EdgarXbrlClient.get_available_quarterly_periods()), claims
        are built for that SPECIFIC historical quarter. Otherwise, the
        most recent available quarter is used -- the normal behaviour
        for a one-off "current" analysis run.
        """
        logger.info(f"Fundamental Agent analyzing {ticker} for {time_period}...")
        claims = self._build_fundamental_claims(ticker, time_period, target_period_end=target_period_end)
        logger.info(f"Fundamental Agent produced {len(claims)} claims for {ticker}.")
        return claims
