"""
verifyfin.agents.macro_agent
==============================
Macro Agent (Proposal Objective 2.3): produces structured, source-cited
ClaimObjects about the macroeconomic environment (Federal Funds Rate,
CPI, unemployment, GDP), sourced from the Federal Reserve's official
FRED API.

Follows the same "LLM phrases, code sources" design as FundamentalAgent:
every numeric value and its SourceReference comes directly from
FredClient's structured output, deterministic and Python-side.

Multi-period support (added for the evaluation harness's "5 tickers x
3 time periods" design): analyze() accepts an optional target_date. If
given, the LATEST FRED observation DATED ON OR BEFORE that date is used
for each series instead of the series' latest value -- FRED's full time
series is already fetched by FredClient.get_series(), so this reuses
existing data rather than requiring a new API call per historical
scenario.

NO LOOK-AHEAD (changed in v3): earlier versions used the observation
CLOSEST to the target date, which for a quarter ending 28 March picked the
observation dated 1 April (a figure describing April, published after the
quarter ended). An analysis "as of" a period end must not use data from
after it, so only observations dated on or before the target are eligible.

Honest, documented limitations: (1) unlike Fundamental Agent claims (which
can be pinned to an exact historical quarter via XBRL), the observation
date can be up to ~1 month before the target_date because FRED series
update monthly or quarterly; the real observation date is always included
in each claim's excerpt so this is never hidden. (2) "Dated on or before"
is not "published on or before": a monthly figure dated the 1st describes
the whole month and is released later, and FRED returns the latest revised
values. A fully point-in-time analysis would need ALFRED release vintages.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from verifyfin.data_pipeline.fred_client import FredClient
from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, SourceReference
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SERIES_LABELS = {
    "FEDFUNDS": "the Federal Funds Rate",
    "CPIAUCSL": "the Consumer Price Index (CPI)",
    "UNRATE": "the U.S. unemployment rate",
    "GDP": "U.S. Gross Domestic Product (GDP)",
}

_SYSTEM_PROMPT = (
    "You are a precise macroeconomic analyst assistant. You will be given a list of "
    "macroeconomic indicator values already retrieved from the Federal Reserve's official "
    "FRED database. Your ONLY job is to phrase each figure as a single, clear, factual "
    "sentence for a financial research report.\n"
    "Rules:\n"
    "1. Use ONLY the numbers given to you -- never invent, estimate, or adjust any figure.\n"
    "2. Do not add any information not present in the data given.\n"
    "3. Do not use words like 'approximately' and do not round the numbers.\n"
    "4. Always state the observation date given.\n"
    "5. Reply with ONLY a JSON array, no other text, in this exact format:\n"
    '[{"series_id": "<series_id>", "claim_text": "<sentence>"}, ...]\n'
    "6. Include exactly one entry per series you were given."
)


class MacroAgent:
    def __init__(self, config_path: Optional[str] = None, llm_client: Optional[GroqClient] = None):
        self.config_path = config_path
        self._fred = FredClient(config_path)
        self._llm = llm_client if llm_client is not None else GroqClient(config_path)

    # ---------- data retrieval ----------

    def _get_series_data(self, target_date: Optional[str]) -> dict[str, dict]:
        """
        Returns {series_id: {"latest_value", "latest_date"}} for every
        configured FRED series. If target_date is given, picks the latest
        observation dated ON OR BEFORE that date from each series' full
        history (never one after it) instead of the series' true latest value.
        """
        if target_date is None:
            return self._fred.get_macro_snapshot()

        result = {}
        for series_id in self._fred.default_series:
            try:
                series = self._fred.get_series(series_id)
            except Exception as e:
                logger.warning(f"Skipping unavailable series {series_id}: {e}")
                continue

            idx = self._latest_observation_index_on_or_before(series["dates"], target_date)
            if idx is None:
                logger.warning(f"No {series_id} observation dated on or before {target_date}; series skipped.")
                continue

            result[series_id] = {
                "series_id": series_id,
                "latest_value": series["values"][idx],
                "latest_date": series["dates"][idx],
            }
        return result

    @staticmethod
    def _latest_observation_index_on_or_before(dates: list[str], target_date: str) -> Optional[int]:
        """Index of the most recent observation dated <= target_date (ISO strings compare chronologically); None if there is none."""
        best_idx, best_date = None, None
        for i, d in enumerate(dates):
            if d <= target_date and (best_date is None or d > best_date):
                best_idx, best_date = i, d
        return best_idx

    # ---------- LLM phrasing step ----------

    def _phrase_series_as_claims(self, series_data: dict[str, dict]) -> dict[str, str]:
        payload_for_llm = [
            {
                "series_id": series_id,
                "label": _SERIES_LABELS.get(series_id, series_id),
                "latest_value": data["latest_value"],
                "latest_date": data["latest_date"],
            }
            for series_id, data in series_data.items()
        ]

        user_prompt = f"Macroeconomic indicators:\n{json.dumps(payload_for_llm, indent=2)}"
        raw_response = self._llm.complete(user_prompt=user_prompt, system_prompt=_SYSTEM_PROMPT)
        phrased = self._parse_llm_claims(raw_response)

        result = {}
        for series_id, data in series_data.items():
            if series_id in phrased:
                result[series_id] = phrased[series_id]
            else:
                logger.warning(f"LLM did not return a phrased claim for '{series_id}'; using deterministic fallback.")
                label = _SERIES_LABELS.get(series_id, series_id)
                result[series_id] = f"{label} was {data['latest_value']} as of {data['latest_date']}, per FRED."
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
                f"Failed to parse LLM claims JSON ({e}); falling back to deterministic phrasing for all series."
            )
            return {}

        result = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "series_id" in item and "claim_text" in item:
                    result[item["series_id"]] = item["claim_text"]
        return result

    # ---------- claim construction ----------

    def _build_macro_claims(self, ticker: str, time_period: str, target_date: Optional[str]) -> list[ClaimObject]:
        snapshot = self._get_series_data(target_date)

        if not snapshot:
            logger.warning("No macro data available from FRED; no macro claims produced.")
            return []

        phrased_claims = self._phrase_series_as_claims(snapshot)

        claims = []
        for series_id, data in snapshot.items():
            claims.append(
                ClaimObject(
                    claim_text=phrased_claims[series_id],
                    claim_category=ClaimCategory.MACRO,
                    produced_by=AgentRole.MACRO,
                    source_reference=SourceReference(
                        source_type="fred",
                        document_id=series_id,
                        excerpt=f"{series_id}: {data['latest_value']} (as of {data['latest_date']})",
                    ),
                    confidence_score=0.95,
                    ticker=ticker.upper(),
                    time_period=time_period,
                )
            )
        return claims

    # ---------- public API ----------

    def analyze(self, ticker: str, time_period: str, target_date: Optional[str] = None) -> list[ClaimObject]:
        """
        Produces all Macro Agent claims.

        If target_date is given (a "YYYY-MM-DD" string, typically a
        Fundamental Agent scenario's period_end), each series uses its
        latest observation dated on or before that date (never a later
        one) rather than the true latest value -- an approximation, since
        FRED series don't necessarily have an observation on the exact
        date requested (see module docstring). Otherwise, the true latest
        value is used, the normal behaviour for a one-off "current"
        analysis run.
        """
        logger.info(f"Macro Agent analyzing macroeconomic environment for {time_period}...")
        claims = self._build_macro_claims(ticker, time_period, target_date=target_date)
        logger.info(f"Macro Agent produced {len(claims)} claims.")
        return claims
