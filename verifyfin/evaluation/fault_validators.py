"""
verifyfin.evaluation.fault_validators
=======================================
Makes fault-injection ground truth trustworthy: a corruption is only used if
it is PROVABLY false.

Why this matters: some "corruptions" can accidentally produce a TRUE claim.
    * shifting the date of a flat series (unemployment 4.1% in July AND August)
    * swapping basic and diluted EPS when the two are equal
    * shifting a fiscal period when the value happens to repeat
A true claim counted as a "missed corruption" would bias the detection rate
DOWN and make the verifier look worse than it is. So for the operators in
fault_injection.NEEDS_VALIDATION, the corrupted statement is checked against
independent truth:
    * fundamentals  -> XBRL facts for the scenario's period (official, structured)
    * macro         -> the FRED observation chunks already indexed in the scenario's knowledge base
and is kept only if the claimed value differs CLEARLY (> 1% and > 0.05 absolute) from
the true value. Ambiguous cases (close values, missing truth) are dropped, not kept.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

_FRED_CHUNK = re.compile(r"^\s*(\w+) observation: (-?[\d.]+) on (\d{4}-\d{2}-\d{2})\s*$")


def clearly_different(claimed: float, true: float) -> bool:
    return abs(claimed - true) > max(0.01 * abs(true), 0.05)


def fred_value(store, ticker: str, series: str, date: str) -> Optional[float]:
    """True value of a FRED series at a date, read from the observation chunks indexed in the knowledge base."""
    try:
        hits = store.query(f"{series} observation on {date}", ticker=ticker, top_k=12, where_document_contains=f" on {date}")
    except Exception:
        return None
    for h in hits:
        m = _FRED_CHUNK.match(h["text"])
        if m and m.group(1) == series and m.group(3) == date:
            return float(m.group(2))
    return None


def make_validator(*, ticker: str, period_end: str, store, xbrl_truth: Callable[[str, str], Optional[dict]],
                   known_period_ends: list[str]) -> Callable[[dict, str, str, dict], Optional[bool]]:
    """
    Returns validator(base, operator, new_text, meta) -> True (provably false) / False (not provably false) / None (cannot tell).
    xbrl_truth(ticker, period_end) -> {concept: {"value": number, ...}} or None.
    """
    def validator(base: dict, op: str, new_text: str, meta: dict) -> Optional[bool]:
        if op == "label_swap":
            truth = xbrl_truth(ticker, period_end)
            if not truth or meta.get("claimed_value") is None:
                return None
            entry = truth.get(meta["to_concept"])
            if not entry:
                return None
            return clearly_different(float(meta["claimed_value"]), float(entry["value"]))
        if op == "period_shift_year":
            concept, claimed = meta.get("concept"), meta.get("claimed_value")
            if concept is None or claimed is None:
                return None
            values = []
            for pe in known_period_ends:
                if pe == period_end:      # the claim's OWN period legitimately holds this value; only OTHER periods can make the shift 'true'
                    continue
                t = xbrl_truth(ticker, pe)
                if t and t.get(concept):
                    values.append(float(t[concept]["value"]))
            if not values:
                return None
            return not any(not clearly_different(float(claimed), v) for v in values)
        if op in ("date_shift_month", "date_shift_year", "series_swap"):
            series, date, claimed = meta.get("series"), meta.get("date"), meta.get("claimed_value")
            if not series or not date or claimed is None:
                return None
            true = fred_value(store, ticker, series, date)
            if true is None:
                return None
            return clearly_different(float(claimed), true)
        return None

    return validator
