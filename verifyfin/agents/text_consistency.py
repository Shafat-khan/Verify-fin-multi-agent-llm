"""
verifyfin.agents.text_consistency
====================================
Deterministic "does the SENTENCE say what the EVIDENCE says?" check.

WHY THIS EXISTS
    The fault-injection experiment (faculty feedback point 1) showed that the
    Verification Agent catches ~88% of errors in the CITED VALUE but only
    ~13% of errors in the SENTENCE when the cited value is right (a wrong
    number or a wrong date in the claim text). The reason is structural: the
    verifier takes the number it searches for from the claim's source excerpt
    and never reads the number in the claim text itself. In VERIFY-FIN the
    numbers are supplied by code and the LLM only phrases them, so this gap is
    the one place a phrasing error could pass unnoticed.

WHAT IT DOES (no LLM calls, no embeddings)
    1. Pulls every quantity out of the claim text ("$116,017,000,000", "$2.3",
       "4.3", "$116.0 billion", "8.5%") together with how precisely it was
       written (decimals shown, scale word).
    2. Pulls every full date out of the claim text (ISO "2026-07-01" and
       "June 27, 2026").
    3. Compares them with the single value and date held in the source excerpt.

    number: OK if some quantity in the sentence equals the excerpt value
            ROUNDED TO THE PRECISION THE SENTENCE USES (so "$116 billion" is
            accepted for 116,017,000,000, but "$117.2 billion" is not).
            MISMATCH if the sentence states quantities and none matches.
            NOT_STATED if the sentence states no quantity (nothing to check).
    date:   OK if the excerpt's date appears among the sentence's dates.
            MISMATCH if the sentence states dates and the excerpt's is absent.
            NOT_STATED if either side has no date.

KNOWN LIMITS (state them in the thesis)
    * Only the excerpt's one value and one date are checked; a sentence that
      states the right number but a wrong *label* is not caught here (that is
      the attribution family, which the retrieval step partly catches).
    * Relative dates ("last quarter") and month-only dates are not parsed.
    * Rounded figures accept a corruption smaller than their own rounding step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_LONG_DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.I)
_QTY = re.compile(
    r"(?<![\w.])(-?)\$?\s?(\d[\d,]*(?:\.\d+)?)(?:\s*(%|percent|trillion|billion|million|thousand|bn|tn|mm|[BMKT])(?![A-Za-z]))?",
    re.I)
_SCALE = {"trillion": 1e12, "tn": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
          "million": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


@dataclass(frozen=True)
class Quantity:
    display: float      # the number as written
    decimals: int       # digits after the decimal point as written
    scale: float        # multiplier from a scale word (1 if none)
    percent: bool

    @property
    def value(self) -> float:
        return self.display * self.scale


@dataclass(frozen=True)
class TextCheck:
    number: str   # "ok" | "mismatch" | "not_stated"
    date: str     # "ok" | "mismatch" | "not_stated"

    @property
    def consistent(self) -> bool:
        return self.number != "mismatch" and self.date != "mismatch"

    @property
    def detail(self) -> str:
        return f"number={self.number}, date={self.date}"


def extract_dates(text: str) -> set[str]:
    """Every full calendar date in the text, normalised to ISO (YYYY-MM-DD)."""
    out = {f"{y}-{m}-{d}" for y, m, d in _ISO.findall(text)}
    for mon, day, year in _LONG_DATE.findall(text):
        out.add(f"{int(year):04d}-{_MONTHS[mon.lower()]:02d}-{int(day):02d}")
    return out


def extract_quantities(text: str) -> list[Quantity]:
    """Quantities stated in the text, excluding anything that is part of a date, a bare year or a quarter tag."""
    spans = [m.span() for m in _ISO.finditer(text)] + [m.span() for m in _LONG_DATE.finditer(text)]
    out: list[Quantity] = []
    for m in _QTY.finditer(text):
        if any(s <= m.start(2) < e for s, e in spans):
            continue
        digits, suffix = m.group(2).rstrip(","), (m.group(3) or "").lower()
        before = text[max(0, m.start() - 2):m.start()]
        if before.upper().endswith("Q") or before.upper().endswith("FY"):
            continue
        dollar = "$" in text[m.start():m.start(2)]
        is_year = (re.fullmatch(r"(19|20)\d{2}", digits) is not None and not dollar and not suffix)
        if is_year:
            continue
        try:
            display = float(digits.replace(",", ""))
        except ValueError:
            continue
        if m.group(1) == "-":
            display = -display
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        percent = suffix in ("%", "percent")
        out.append(Quantity(display, decimals, 1.0 if percent else _SCALE.get(suffix, 1.0), percent))
    return out


def quantity_matches(q: Quantity, value: float) -> bool:
    """True if `value` rounded to the precision the sentence used equals the number written."""
    scaled = value / q.scale
    half_step = 0.5 * 10 ** (-q.decimals)
    return abs(scaled - q.display) <= half_step * (1 + 1e-9) + 1e-12


def check_text_against_excerpt(claim_text: str, excerpt_value: Optional[float], excerpt_date: Optional[str]) -> TextCheck:
    quantities = extract_quantities(claim_text)
    if excerpt_value is None or not quantities:
        number = "not_stated"
    else:
        number = "ok" if any(quantity_matches(q, excerpt_value) for q in quantities) else "mismatch"

    dates = extract_dates(claim_text)
    if not excerpt_date or not dates:
        date = "not_stated"
    else:
        date = "ok" if excerpt_date in dates else "mismatch"
    return TextCheck(number=number, date=date)
