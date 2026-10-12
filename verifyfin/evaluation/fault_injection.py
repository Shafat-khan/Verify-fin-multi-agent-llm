"""
verifyfin.evaluation.fault_injection
=======================================
Fault injection ("mutation testing") for the Verification Agent
(faculty feedback point 1).

THE PROBLEM THIS ADDRESSES
    The human evaluation measures VERIFY-FIN on the claims the pipeline
    naturally produced, and those contain almost no errors to catch
    (12 of 171). The verifier is therefore both the component under test
    and, in the post-hoc rates, the instrument that measures the outcome,
    and there are too few real errors to test it properly.

THE IDEA
    Take claims that humans judged SUPPORTED, corrupt them in controlled
    ways, and measure
        * how many corruptions the verifier catches (detection / flag rate),
        * how many UNcorrupted claims it wrongly flags (false-flag rate).
    Ground truth is known BY CONSTRUCTION: a corrupted claim is false, an
    original is true. This is the same idea as mutation testing in software
    engineering (DeMillo, Lipton and Sayward, 1978; Jia and Harman, 2011),
    the perturbation tests of CheckList (Ribeiro et al., 2020), and the
    claim-mutation used to build refuted claims in FEVER (Thorne et al., 2018).

TWO INJECTION MODES (they probe different parts of the verifier)
    "source"   -- the CITED VALUE is wrong: both the claim text and the
                  source excerpt carry the corrupted value/date (an upstream
                  data fault, e.g. wrong period or wrong XBRL tag). The
                  verifier's numeric match looks for the (wrong) value.
    "phrasing" -- the data was right but the SENTENCE is wrong: the claim
                  text is corrupted while the excerpt keeps the true value
                  (an LLM mis-phrasing). The verifier takes its expected
                  number from the excerpt, so this tests whether the cosine
                  similarity of the sentence alone notices the error.
    "label"    -- the sentence attributes a true value to the wrong item
                  (revenue vs net income; unemployment vs fed funds). The
                  excerpt holds only value and date, so mode is irrelevant.

VALIDITY: a corruption must be FALSE, never accidentally true (e.g. shifting
the date of a series that happens to be flat, or swapping basic and diluted
EPS when they are equal). Operators flagged in NEEDS_VALIDATION are accepted
only if a caller-supplied validator confirms, from XBRL facts or the FRED
chunks in the knowledge base, that the corrupted statement is false.

EXCERPT RECONSTRUCTION: the per-claim records do not store the source
excerpt, so it is reconstructed from the claim text in the same format the
analyst agents produce. The runner VERIFIES this reconstruction by re-scoring
the uncorrupted claims and comparing with the original pipeline run.
"""

from __future__ import annotations

import math
import random
import re
from calendar import monthrange
from datetime import date
from typing import Callable, Optional, Sequence

_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)(\s*(?:billion|million|thousand))?", re.I)
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
_MONTH_DATE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORD_MULT = {"billion": 1e9, "million": 1e6, "thousand": 1e3}

_NUMERIC_LABELS = [  # (pattern, concept, replacement label, replacement concept)
    (re.compile(r"total revenue", re.I), "revenue", "net income", "net_income"),
    (re.compile(r"net income", re.I), "net_income", "total revenue", "revenue"),
    (re.compile(r"basic earnings per share", re.I), "eps_basic", "diluted earnings per share", "eps_diluted"),
    (re.compile(r"diluted earnings per share", re.I), "eps_diluted", "basic earnings per share", "eps_basic"),
    (re.compile(r"total assets", re.I), "total_assets", "total liabilities", "total_liabilities"),
    (re.compile(r"total liabilities", re.I), "total_liabilities", "total assets", "total_assets"),
    (re.compile(r"stockholders[\u2019']\s*equity", re.I), "stockholders_equity", "total liabilities", "total_liabilities"),
    (re.compile(r"operating income", re.I), "operating_income", "net income", "net_income"),
]
_SERIES = [  # (pattern, series id, replacement label, replacement series id)
    (re.compile(r"Federal Funds Rate", re.I), "FEDFUNDS", "U.S. unemployment rate", "UNRATE"),
    (re.compile(r"(?:U\.S\. )?unemployment rate", re.I), "UNRATE", "Federal Funds Rate", "FEDFUNDS"),
    (re.compile(r"Consumer Price Index(?: \(CPI\))?", re.I), "CPIAUCSL", "U.S. unemployment rate", "UNRATE"),
    (re.compile(r"(?:U\.S\. )?Gross Domestic Product(?: \(GDP\))?", re.I), "GDP", "Federal Funds Rate", "FEDFUNDS"),
]

NEEDS_VALIDATION = {"label_swap", "series_swap", "period_shift_year", "date_shift_month", "date_shift_year"}

OPERATORS = {
    "numerical": [
        ("scale_1pct", "magnitude"), ("scale_5pct", "magnitude"), ("scale_25pct", "magnitude"), ("scale_10x", "magnitude"),
        ("digit_transpose", "typo"), ("digit_noise_1", "typo"), ("digit_noise_3", "typo"),
        ("label_swap", "attribution"), ("period_shift_year", "temporal"),
    ],
    "macro": [
        ("scale_1pct", "magnitude"), ("scale_5pct", "magnitude"), ("scale_25pct", "magnitude"),
        ("digit_noise_1", "typo"),
        ("date_shift_month", "temporal"), ("date_shift_year", "temporal"), ("series_swap", "attribution"),
    ],
}
_SCALE = {"scale_1pct": 1.01, "scale_5pct": 1.05, "scale_25pct": 1.25, "scale_10x": 10.0}


# ----------------------------------------------------------------- number handling

def format_like(value: float, decimals: int, commas: bool) -> str:
    return f"{value:,.{decimals}f}" if commas else f"{value:.{decimals}f}"


def parse_amount(text: str) -> Optional[dict]:
    """First dollar amount in a fundamental claim: digit span, formatting style, value in dollars."""
    m = _MONEY.search(text)
    if not m:
        return None
    digits = m.group(1)
    word = (m.group(2) or "").strip().lower()
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    display = float(digits.replace(",", ""))
    return {"start": m.start(1), "end": m.end(1), "digits": digits, "decimals": decimals, "commas": "," in digits,
            "display": display, "mult": _WORD_MULT.get(word, 1.0)}


def _numeric_string(value: float) -> str:
    return str(int(round(value))) if abs(value - round(value)) <= 1e-9 * max(1.0, abs(value)) else repr(value)


def reconstruct_numerical_excerpt(text: str, period_end: str) -> Optional[str]:
    """Rebuild the source excerpt the Fundamental Agent attaches: 'TAG: value UNIT (period X to END)'."""
    amt = parse_amount(text)
    if amt is None:
        return None
    unit = "USD/shares" if "per share" in text.lower() else "USD"
    return f"Reconstructed: {_numeric_string(amt['display'] * amt['mult'])} {unit} (period N/A to {period_end})"


def find_macro_value(text: str) -> Optional[dict]:
    """The reported value in a macro claim: the last number that is not part of an ISO date."""
    spans = [m.span() for m in _ISO.finditer(text)]
    best = None
    for m in _NUM.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        best = m
    if best is None:
        return None
    tok = best.group(0)
    return {"start": best.start(), "end": best.end(), "token": tok.replace(",", ""),
            "decimals": len(tok.split(".")[1]) if "." in tok else 0, "commas": "," in tok}


def reconstruct_macro_excerpt(text: str) -> Optional[str]:
    """Rebuild 'SERIES: value (as of DATE)' from the claim text."""
    v = find_macro_value(text)
    dates = _ISO.findall(text)
    if v is None or not dates:
        return None
    return f"Reconstructed: {v['token']} (as of {dates[-1]})"


def reconstruct_excerpt(category: str, text: str, period_end: Optional[str]) -> Optional[str]:
    if category == "numerical":
        return reconstruct_numerical_excerpt(text, period_end or "N/A")
    if category == "macro":
        return reconstruct_macro_excerpt(text)
    return None


def numeric_concept(text: str) -> Optional[str]:
    for pat, concept, _, _ in _NUMERIC_LABELS:
        if pat.search(text):
            return concept
    return None


def macro_series(text: str) -> Optional[str]:
    for pat, sid, _, _ in _SERIES:
        if pat.search(text):
            return sid
    return None


# ----------------------------------------------------------------- operators

def _splice(text: str, start: int, end: int, new: str) -> str:
    return text[:start] + new + text[end:]


def _mutate_digits(digits: str, rng: random.Random, k: int) -> Optional[str]:
    pos = [i for i, c in enumerate(digits) if c.isdigit()]
    if len(pos) < k:
        return None
    chosen = rng.sample(pos, k)
    chars = list(digits)
    for i in chosen:
        old = chars[i]
        options = [d for d in "0123456789" if d != old and not (i == pos[0] and d == "0")]
        chars[i] = rng.choice(options)
    return "".join(chars)


def _transpose(digits: str, rng: random.Random) -> Optional[str]:
    pos = [i for i, c in enumerate(digits) if c.isdigit()]
    cands = []
    for a, b in zip(pos, pos[1:]):
        if digits[a] != digits[b] and not (a == pos[0] and digits[b] == "0"):
            cands.append((a, b))
    if not cands:
        return None
    a, b = rng.choice(cands)
    chars = list(digits)
    chars[a], chars[b] = chars[b], chars[a]
    return "".join(chars)


def _shift_iso(iso: str, months: int = 0, years: int = 0) -> str:
    y, m, d = (int(x) for x in iso.split("-"))
    total = (y + years) * 12 + (m - 1) + months
    ny, nm = divmod(total, 12)
    nm += 1
    return date(ny, nm, min(d, monthrange(ny, nm)[1])).isoformat()


def _numeric_value_op(base: dict, op: str, rng: random.Random):
    text = base["text"]
    amt = parse_amount(text)
    if amt is None:
        return None
    new_digits = None
    if op in _SCALE:
        new_display = amt["display"] * _SCALE[op]
        new_digits = format_like(new_display, amt["decimals"], amt["commas"])
        if abs(new_display - amt["display"]) / max(abs(amt["display"]), 1e-12) < 0.004 or new_digits == amt["digits"]:
            return None
        meta = {"factor": _SCALE[op]}
    elif op == "digit_transpose":
        new_digits = _transpose(amt["digits"], rng)
        meta = {}
    elif op in ("digit_noise_1", "digit_noise_3"):
        new_digits = _mutate_digits(amt["digits"], rng, 1 if op.endswith("1") else 3)
        meta = {}
    else:
        return None
    if new_digits is None or new_digits == amt["digits"]:
        return None
    return _splice(text, amt["start"], amt["end"], new_digits), base.get("period_end"), meta


def _numeric_label_swap(base: dict, rng: random.Random):
    text = base["text"]
    amt = parse_amount(text)
    for pat, concept, repl_label, repl_concept in _NUMERIC_LABELS:
        m = pat.search(text)
        if m:
            return (_splice(text, m.start(), m.end(), repl_label), base.get("period_end"),
                    {"from_concept": concept, "to_concept": repl_concept,
                     "claimed_value": (amt["display"] * amt["mult"]) if amt else None})
    return None


def _numeric_period_shift(base: dict, rng: random.Random):
    text = base["text"]
    m = _MONTH_DATE.search(text)
    pe = base.get("period_end")
    if not m or not pe:
        return None
    new_pe = f"{int(pe[:4]) - 1}{pe[4:]}"
    new_text = _splice(text, m.start(3), m.end(3), str(int(m.group(3)) - 1))
    amt = parse_amount(text)
    return new_text, new_pe, {"concept": numeric_concept(text), "claimed_value": (amt["display"] * amt["mult"]) if amt else None,
                              "new_period_end": new_pe}


def _macro_value_op(base: dict, op: str, rng: random.Random):
    text = base["text"]
    v = find_macro_value(text)
    if v is None:
        return None
    old = float(v["token"])
    if op in _SCALE:
        new_tok = format_like(old * _SCALE[op], v["decimals"], False)
        if abs(float(new_tok) - old) / max(abs(old), 1e-12) < 0.004:
            return None
        meta = {"factor": _SCALE[op]}
    elif op == "digit_noise_1":
        new_tok = _mutate_digits(v["token"], rng, 1)
        meta = {}
    else:
        return None
    if new_tok is None or new_tok == v["token"]:
        return None
    dates = _ISO.findall(text)
    meta.update({"series": macro_series(text), "date": dates[-1] if dates else None, "claimed_value": float(new_tok)})
    return _splice(text, v["start"], v["end"], new_tok), base.get("period_end"), meta


def _macro_date_shift(base: dict, op: str, rng: random.Random):
    text = base["text"]
    m = _ISO.search(text)
    v = find_macro_value(text)
    if not m or v is None:
        return None
    new = _shift_iso(m.group(0), months=1) if op == "date_shift_month" else _shift_iso(m.group(0), years=-1)
    return _splice(text, m.start(), m.end(), new), base.get("period_end"), {
        "series": macro_series(text), "date": new, "claimed_value": float(v["token"])}


def _macro_series_swap(base: dict, rng: random.Random):
    text = base["text"]
    v = find_macro_value(text)
    dates = _ISO.findall(text)
    for pat, sid, repl_label, repl_sid in _SERIES:
        m = pat.search(text)
        if m and v is not None and dates:
            return _splice(text, m.start(), m.end(), repl_label), base.get("period_end"), {
                "from_series": sid, "series": repl_sid, "date": dates[-1], "claimed_value": float(v["token"])}
    return None


def apply_operator(base: dict, op: str, rng: random.Random):
    cat = base["category"]
    if cat == "numerical":
        if op == "label_swap":
            return _numeric_label_swap(base, rng)
        if op == "period_shift_year":
            return _numeric_period_shift(base, rng)
        return _numeric_value_op(base, op, rng)
    if cat == "macro":
        if op in ("date_shift_month", "date_shift_year"):
            return _macro_date_shift(base, op, rng)
        if op == "series_swap":
            return _macro_series_swap(base, rng)
        return _macro_value_op(base, op, rng)
    return None


# ----------------------------------------------------------------- variant generation

Validator = Callable[[dict, str, str, dict], Optional[bool]]


def generate_variants(base: dict, *, seed: int = 42, validator: Optional[Validator] = None,
                      operators: Optional[Sequence[str]] = None) -> tuple[list[dict], dict]:
    """
    All corrupted variants of one base claim.

    base: dict with claim_id, ticker, time_period, period_end, category ('numerical'|'macro'), text, excerpt.
    validator(base, operator, new_text, meta): True if the corrupted statement is FALSE (confirmed), else False/None;
    only consulted for operators in NEEDS_VALIDATION. Returns (variants, drop_counts).
    """
    out, drops = [], {"inapplicable": 0, "not_false": 0, "unvalidated": 0}
    for op, family in OPERATORS.get(base["category"], []):
        if operators and op not in operators:
            continue
        rng = random.Random(f"{seed}|{base['claim_id']}|{op}")
        res = apply_operator(base, op, rng)
        if res is None or res[0] == base["text"]:
            drops["inapplicable"] += 1
            continue
        new_text, new_pe, meta = res
        if validator is not None and op in NEEDS_VALIDATION:
            verdict = validator(base, op, new_text, meta)
            if verdict is None:
                drops["unvalidated"] += 1
                continue
            if verdict is not True:
                drops["not_false"] += 1
                continue
        modes = ["label"] if family == "attribution" else ["phrasing", "source"]
        for mode in modes:
            excerpt = (reconstruct_excerpt(base["category"], new_text, new_pe)
                       if mode == "source" else base["excerpt"])
            out.append({
                "base_id": base["claim_id"], "ticker": base["ticker"], "time_period": base["time_period"],
                "category": base["category"], "operator": op, "family": family, "mode": mode,
                "text": new_text, "excerpt": excerpt, "period_end": new_pe, "meta": meta,
            })
    return out, drops


# ----------------------------------------------------------------- aggregation

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cluster_bootstrap_rate(base_ids: Sequence[str], flags: Sequence[bool], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% percentile interval for a rate when variants of the same base claim are NOT independent:
    resample whole base claims (with all their variants) with replacement."""
    import numpy as np

    per: dict[str, list[int]] = {}
    for b, f in zip(base_ids, flags):
        c = per.setdefault(b, [0, 0])
        c[0] += 1
        c[1] += int(bool(f))
    if not per:
        return float("nan"), float("nan")
    arr = np.array(list(per.values()), dtype=float)
    k = len(arr)
    idx = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    n, hit = arr[idx, 0].sum(1), arr[idx, 1].sum(1)
    rate = hit / n
    lo, hi = np.percentile(rate, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(rows: Sequence[dict], keys: Sequence[str], n_boot: int = 2000) -> list[dict]:
    """
    Group rows (each: base_id, verdict, score, plus grouping keys) and compute, per group: counts per verdict,
    flag rate (verdict != 'grounded' = caught or sent to Debate), hard-block rate, Wilson and cluster-bootstrap intervals.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        n = len(rs)
        grounded = sum(1 for r in rs if r["verdict"] == "grounded")
        challenged = sum(1 for r in rs if r["verdict"] == "challenged")
        blocked = sum(1 for r in rs if r["verdict"] == "blocked")
        flagged = n - grounded
        wl, wh = wilson(flagged, n)
        cl, ch = cluster_bootstrap_rate([r["base_id"] for r in rs], [r["verdict"] != "grounded" for r in rs], n_boot)
        out.append({**dict(zip(keys, g)), "n": n, "grounded": grounded, "challenged": challenged, "blocked": blocked,
                    "flag_rate": flagged / n, "wilson_lo": wl, "wilson_hi": wh, "cluster_lo": cl, "cluster_hi": ch,
                    "block_rate": blocked / n, "mean_score": sum(r["score"] for r in rs) / n})
    return out


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie): the Mann-Whitney statistic / (n_pos * n_neg)."""
    from verifyfin.evaluation.stats_utils import rankdata_average

    if not positives or not negatives:
        return float("nan")
    ranks = rankdata_average(list(positives) + list(negatives))
    r_pos = sum(ranks[: len(positives)])
    n1, n2 = len(positives), len(negatives)
    return (r_pos - n1 * (n1 + 1) / 2) / (n1 * n2)


def paired_clean_higher(pairs: Sequence[tuple[float, float]]) -> dict:
    """pairs: (clean_score, corrupted_score) for the same base claim. How often is the clean version scored higher?"""
    if not pairs:
        return {"n": 0, "higher": float("nan"), "equal": float("nan"), "lower": float("nan"), "mean_drop": float("nan")}
    n = len(pairs)
    return {"n": n, "higher": sum(c > x for c, x in pairs) / n, "equal": sum(c == x for c, x in pairs) / n,
            "lower": sum(c < x for c, x in pairs) / n, "mean_drop": sum(c - x for c, x in pairs) / n}


def threshold_sweep(control_scores: Sequence[float], corrupted_scores: Sequence[float],
                    thresholds: Sequence[float]) -> list[dict]:
    """For each grounded-threshold t: a claim is FLAGGED when score < t. Returns detection (corrupted flagged) and false-flag rates."""
    rows = []
    for t in thresholds:
        det = sum(s < t for s in corrupted_scores) / len(corrupted_scores) if corrupted_scores else float("nan")
        ff = sum(s < t for s in control_scores) / len(control_scores) if control_scores else float("nan")
        rows.append({"threshold": t, "detection": det, "false_flag": ff, "youden_j": det - ff})
    return rows
