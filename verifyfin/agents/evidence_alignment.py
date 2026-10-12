"""
verifyfin.agents.evidence_alignment
======================================
Deterministic checks on WHICH passage a claim was matched to (v3 of the Verification Agent).

WHY THIS EXISTS
    Two kinds of error survive the v1/v2 verifier, and both were seen on REAL claims (not only injected faults):

      * right NUMBER, wrong DATE.  The Federal Funds Rate was 3.64 in January, March and April 2026. A claim
        "3.64 on 2026-04-01" was matched to the chunk "3.64 on 2026-03-01" and passed (score 0.94) although the
        April chunk existed in the knowledge base. The number-matching step only looks at the top-8 semantic
        hits, and the date preference is a tie-break among them, so a date-correct chunk outside that pool is
        never seen.
      * right NUMBER, wrong LABEL / SOURCE. "unemployment 4.1 on 2026-07-01" was matched to a bank's
        forecast table that happens to contain "4.1", not to the "UNRATE observation: 4.1 on 2026-07-01" chunk.

    The checks below look at the matched passage itself. They use no LLM and no embeddings.

WHAT IT PROVIDES (all pure functions, unit-tested)
    contains_number      a number appears as a whole token (4.1 is not found inside 14.1 or 4.12)
    FRED records         "SERIES observation: value on DATE": parse, find the exact (value, date) record
    claim_series         which FRED series a sentence talks about ("unemployment rate" -> UNRATE)
    claim_concepts / chunk_concepts / label_status
                         which financial line item a sentence talks about, and which line items a filing chunk
                         is about. "mismatch" ONLY when the chunk positively shows other line items and not the
                         claimed one; a chunk with no recognisable label is "unknown", never a mismatch.
    period_status        filing tables put the periods in a header row ("June 27,2026 | June 28,2025 | Change")
                         and the numbers in the row after "||". The column holding the claimed number must carry
                         the claimed period end date. "mismatch" if it carries another date, "unknown" if the
                         chunk cannot be aligned (no header, different number of columns).

KNOWN LIMITS (state them in the thesis)
    * Three-month and nine-month columns carry the same date, so a year-to-date number cited as a quarterly one
      is not caught (a duration check would be needed).
    * The line-item vocabulary is a fixed list of English synonyms; an unlisted label gives "unknown".
    * Table chunks that were split away from their header row cannot be aligned.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# ------------------------------------------------------------------ numbers


def contains_number(text: str, variants: Iterable[str]) -> bool:
    """True if any variant occurs as a WHOLE number token (not inside a longer number)."""
    for v in variants:
        if re.search(r"(?<![\d.,])" + re.escape(v) + r"(?!\d|[.,]\d)", text):
            return True
    return False


def _as_floats(variants: Iterable[str]) -> list[float]:
    out = []
    for v in variants:
        try:
            out.append(float(v.replace(",", "")))
        except ValueError:
            pass
    return out


# ------------------------------------------------------------------ FRED observation chunks

_FRED_CHUNK = re.compile(r"^\s*([A-Za-z0-9_]+) observation:\s*(-?\d[\d.]*)\s+on\s+(\d{4}-\d{2}-\d{2})\s*$")

SERIES_PATTERNS = [
    (re.compile(r"federal\s+funds\s+rate", re.I), "FEDFUNDS"),
    (re.compile(r"consumer\s+price\s+index|\bCPI\b", re.I), "CPIAUCSL"),
    (re.compile(r"unemployment\s+rate", re.I), "UNRATE"),
    (re.compile(r"gross\s+domestic\s+product|\bGDP\b", re.I), "GDP"),
]


def parse_fred_chunk(text: str) -> Optional[tuple[str, float, str]]:
    """'FEDFUNDS observation: 3.64 on 2026-04-01' -> ('FEDFUNDS', 3.64, '2026-04-01'); None for any other chunk."""
    m = _FRED_CHUNK.match(text or "")
    if not m:
        return None
    try:
        return m.group(1), float(m.group(2)), m.group(3)
    except ValueError:
        return None


def claim_series(text: str) -> set[str]:
    """FRED series ids the sentence names (empty set if it names none)."""
    return {sid for pat, sid in SERIES_PATTERNS if pat.search(text or "")}


def fred_records(candidates: list[dict], variants: Iterable[str], date: str) -> list[dict]:
    """Candidates that are FRED observation chunks with EXACTLY this date and a value equal to one of the variants."""
    floats = _as_floats(variants)
    out = []
    for c in candidates:
        parsed = parse_fred_chunk(c.get("text", ""))
        if parsed and parsed[2] == date and any(abs(parsed[1] - f) <= 1e-9 for f in floats):
            out.append(c)
    return out


def pick_record(records: list[dict], series_in_text: set[str]) -> Optional[dict]:
    """Prefer a record whose series the sentence names; otherwise the first record (the label check will judge it)."""
    for r in records:
        parsed = parse_fred_chunk(r["text"])
        if parsed and parsed[0] in series_in_text:
            return r
    return records[0] if records else None


def series_status(claim_text: str, chunk_text: str) -> str:
    """ok / mismatch / unknown: does the series named in the sentence match the FRED chunk's series?"""
    parsed = parse_fred_chunk(chunk_text)
    named = claim_series(claim_text)
    if not parsed or not named:
        return "unknown"
    return "ok" if parsed[0] in named else "mismatch"


# ------------------------------------------------------------------ financial line items

# (concept, regex for the SENTENCE, regex for a filing CHUNK)
_I = re.IGNORECASE
_CONCEPTS: list[tuple[str, re.Pattern, re.Pattern]] = [
    ("eps_diluted", re.compile(r"diluted\s+(?:earnings|eps|net\s+income)", _I),
     re.compile(r"\bdiluted\b[^|]{0,60}\bper\b[^|]{0,15}\bshare\b|\bdiluted\s+(?:eps|earnings)\b", _I)),
    ("eps_basic", re.compile(r"basic\s+(?:earnings|eps|net\s+income)", _I),
     re.compile(r"\bbasic\b[^|]{0,60}\bper\b[^|]{0,15}\bshare\b|\bbasic\s+(?:eps|earnings)\b", _I)),
    ("net_income", re.compile(r"net\s+(?:income|earnings)", _I),
     re.compile(r"\bnet\s+(?:income|earnings)\b|\bnet\s+\(?loss\)?\s+income\b", _I)),
    ("revenue", re.compile(r"\b(?:total\s+)?(?:net\s+)?(?:revenues?|sales)\b", _I),
     re.compile(r"\b(?:total\s+)?net\s+sales\b|\b(?:total\s+)?(?:net\s+)?revenues?\b|\bsales\s+to\s+customers\b", _I)),
    ("total_assets", re.compile(r"total\s+assets", _I), re.compile(r"\btotal\s+assets\b", _I)),
    ("total_liabilities", re.compile(r"total\s+liabilities", _I), re.compile(r"\btotal\s+liabilities\b", _I)),
    ("stockholders_equity", re.compile(r"(?:stock|share)holders[’'`]?\s+(?:equity|investment)|total\s+equity", _I),
     re.compile(r"(?:stock|share)holders[’'`]?\s+(?:equity|investment)|\btotal\s+equity\b", _I)),
    ("operating_income", re.compile(r"operating\s+(?:income|profit)|income\s+from\s+operations", _I),
     re.compile(r"\boperating\s+(?:income|profit)\b|\bincome\s+from\s+operations\b", _I)),
]


def claim_concepts(text: str) -> set[str]:
    """Line items the sentence names. 'net income per share' style phrases count as EPS, not net income."""
    found = {c for c, pat, _ in _CONCEPTS if pat.search(text or "")}
    if found & {"eps_basic", "eps_diluted"}:
        found.discard("net_income")
    if "earnings per share" in (text or "").lower() and not found & {"eps_basic", "eps_diluted"}:
        found |= {"eps_basic", "eps_diluted"}
    return found


_BARE_EPS_LABEL = re.compile(r"^(basic|diluted)\s*:?$", _I)
_GENERIC_EPS_LABEL = re.compile(r"^(?:net\s+)?(?:income|earnings)(?:\s*\(loss\))?\s+per\s+(?:common\s+)?share\b", _I)
_XOM_DILUTED = re.compile(r"\bper\s+(?:common\s+)?share\b[^|]{0,40}\b(?:assuming\s+dilution|diluted)\b", _I)
_XOM_BASIC = re.compile(r"\bper\s+(?:common\s+)?share\b", _I)


def _label_concepts(label: str) -> set[str]:
    """Line items named by ONE table row label."""
    label = label.strip()
    bare = _BARE_EPS_LABEL.match(label.strip())
    if bare:                                    # MSFT-style: the row is just 'Basic' / 'Diluted' (heading is lost in chunking)
        return {"eps_" + bare.group(1).lower()}
    if _XOM_DILUTED.search(label):              # 'Earnings (loss) per common share - assuming dilution (dollars)'
        return {"eps_diluted"}
    if _GENERIC_EPS_LABEL.search(label):        # 'Net income per share': EPS, basic or diluted not stated -> compatible with both
        return {"eps_basic", "eps_diluted"}
    found = {c for c, _, pat in _CONCEPTS if pat.search(label)}
    if not found & {"eps_basic", "eps_diluted"} and _XOM_BASIC.search(label) and re.search(r"earnings|income", label, _I):
        found |= {"eps_basic"}
        found.discard("net_income")
    return found


def chunk_concepts(text: str) -> set[str]:
    """
    Line items a chunk shows. For a flattened table ('header cells || row | v | v || row | v ...') ONLY the row labels
    count: the header side holds column titles and section headings ('Revenue:', 'Numerator:') that are not the line
    item of the numbers below them. Text without '||' is read as a whole.
    """
    text = text or ""
    parts = text.split("||")
    if len(parts) < 2:
        return _label_concepts(text)
    out: set[str] = set()
    for row in parts[1:]:
        out |= _label_concepts(row.split("|")[0])
    return out


def label_status(claim_text: str, chunk_text: str) -> str:
    """
    ok       the chunk shows a line item the sentence names
    mismatch the chunk shows other line items, none of them the one the sentence names
    unknown  either side names no recognisable line item (never a mismatch)
    """
    c, k = claim_concepts(claim_text), chunk_concepts(chunk_text)
    if not c or not k:
        return "unknown"
    return "ok" if c & k else "mismatch"


# ------------------------------------------------------------------ filing table columns

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_HEADER_DATE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2}),?\s*(\d{4})\b", re.I)
_VALUE_CELL = re.compile(r"^\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?$")
_CHANGE_CELL = re.compile(r"^(?:%\s*)?(?:change|variance)$", re.I)


def header_date(cell: str) -> Optional[str]:
    """'June 27,2026' / 'Jun. 27, 2026' -> '2026-06-27'."""
    m = _HEADER_DATE.search(cell or "")
    if not m:
        return None
    return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1)[:3].lower()]:02d}-{int(m.group(2)):02d}"


def _cells(part: str) -> list[str]:
    return [c.strip() for c in part.split("|")]


def column_dates(chunk_text: str, variants: Iterable[str]) -> Optional[list[Optional[str]]]:
    """
    Dates of the column(s) holding the claimed number, or None if the chunk cannot be aligned.
    A 'Change' column yields None inside the list (it carries no date).
    """
    parts = (chunk_text or "").split("||")
    if len(parts) < 2:
        return None
    columns: list[Optional[str]] = []
    for cell in _cells(parts[0]):
        d = header_date(cell)
        if d:
            columns.append(d)
        elif _CHANGE_CELL.match(cell):
            columns.append(None)
    if not any(columns):
        return None
    wanted = {v.strip() for v in variants}
    hits: list[Optional[str]] = []
    aligned_any = False
    for row in parts[1:]:
        values = []
        for c in _cells(row)[1:]:
            s = c.replace("$", "").strip()
            if s and _VALUE_CELL.match(s):
                values.append(s)
        if len(values) != len(columns):
            continue
        aligned_any = True
        hits += [columns[i] for i, v in enumerate(values) if v in wanted]
    return hits if aligned_any and hits else None


def period_status(chunk_text: str, variants: Iterable[str], claim_date: Optional[str]) -> str:
    """ok / mismatch / unknown: does the column holding the claimed number carry the claimed period-end date?"""
    if not claim_date:
        return "unknown"
    dates = column_dates(chunk_text, variants)
    if not dates:
        return "unknown"
    return "ok" if claim_date in dates else "mismatch"
