"""
verifyfin.data_pipeline.edgar_xbrl_client
============================================
Client for SEC EDGAR's official XBRL "Company Facts" API
(https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json).

Why this exists (engineering decision, documented for the dissertation):
    yfinance's `.info` property scrapes an unofficial Yahoo Finance
    endpoint that is increasingly aggressively rate-limited/blocked.
    SEC's XBRL Company Facts API, by contrast, is an OFFICIAL,
    documented, stable government API that returns exactly the
    structured financial figures as filed in a company's own 10-K/10-Q.
    This is used as the PRIMARY source of numerical fundamentals for
    the Fundamental Agent's claims.

Companies vary in which exact XBRL tag they report a given concept
under. This client tries a short list of known aliases per concept.

Quarterly vs. year-to-date period selection: "duration" type XBRL
facts (revenue, net income, EPS, operating income) are reported for
MULTIPLE overlapping periods in the same 10-Q -- a Q2 10-Q reports
both its 3-month figure AND its 6-month year-to-date figure, often
under the exact same tag with the exact same period-end date. This
client resolves the ambiguity by preferring the entry whose
(end - start) duration is closest to one quarter (~91 days).

Multi-period support (added for the evaluation harness's "5 tickers x
3 time periods" design, Proposal §2.5): get_latest_fundamentals()
always returns the single most recent quarter, which is fine for
one-off agent runs but wrong for an evaluation that needs 3 genuinely
DISTINCT historical quarters per ticker -- "most recent" silently
drifts as new filings are published, which is exactly the reproducibility
bug that caused a real period-mismatch earlier in development.
get_available_quarterly_periods() + get_fundamentals_for_period() let
the evaluation harness resolve and pin specific historical quarters
once, rather than re-resolving "latest" independently in every run.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from verifyfin.data_pipeline.edgar_client import EdgarClient
from verifyfin.utils.io_utils import resolve_path
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

_CONCEPT_TAG_ALIASES: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        # Financial-sector aliases (added after a real bug found during
        # development: banks like JPM don't report a "Revenues" concept
        # in recent filings the way non-financial companies do -- their
        # income statement is structured around interest income/expense
        # and noninterest income instead. Without these, period
        # resolution silently fell back to the last year JPM ever used
        # a generic "Revenues" tag (2014), a decade-stale result that
        # was never flagged as wrong.
        "RevenuesNetOfInterestExpense",  # the standard "total revenue" tag for many bank holding companies
        "InterestAndDividendIncomeOperating",
    ],
    "net_income": ["NetIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity"],
    "operating_income": ["OperatingIncomeLoss"],
}

_RELEVANT_FORMS = {"10-K", "10-Q"}
_TARGET_QUARTER_DAYS = 91  # approximate length of one fiscal quarter, for duration-based disambiguation
_QUARTER_DURATION_TOLERANCE_DAYS = 15  # a "genuine single quarter" duration window, excludes YTD/annual entries
_CACHE_TTL_HOURS = 24  # matches yfinance_client / fred_client -- a new quarterly filing can appear at any time
_MAX_PERIOD_AGE_DAYS = 730  # ~2 years -- a resolved "recent" quarter older than this is treated as stale/missing,
# not silently used (this is the direct fix for the JPM bug: a broader tag-alias list reduces how often this
# safeguard needs to fire at all, but it exists as a hard backstop regardless of tag coverage completeness)


class XbrlFactsUnavailableError(Exception):
    """Raised when SEC has no XBRL company facts at all for a given ticker (e.g. non-US-GAAP filers)."""


class EdgarXbrlClient:
    def __init__(self, config_path: Optional[str] = None):
        self._edgar = EdgarClient(config_path)  # reuse CIK lookup + rate-limited HTTP
        self.cache_dir = resolve_path("data/raw/edgar_xbrl_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = _CACHE_TTL_HOURS

    # ---------- retrieval ----------

    def _cache_path(self, cik10: str) -> Path:
        return self.cache_dir / f"companyfacts_{cik10}.json"

    def get_company_facts(self, ticker: str) -> dict:
        """
        Returns the full raw XBRL company facts JSON for a ticker,
        cached to disk with a TTL. A cache older than the TTL is
        refetched, since a new quarterly filing can appear on SEC's
        side at any time, and stale cached data describing an older
        quarter than the rest of the pipeline caused a real,
        previously-found bug.
        """
        cik10 = self._edgar.get_cik_for_ticker(ticker)
        cache_path = self._cache_path(cik10)

        if cache_path.exists():
            with open(cache_path, "r") as f:
                payload = json.load(f)
            cached_at = datetime.fromisoformat(payload["cached_at"])
            if datetime.now(timezone.utc) - cached_at <= timedelta(hours=self.cache_ttl_hours):
                logger.debug(f"Using cached XBRL company facts for {ticker} (CIK {cik10}).")
                return payload["data"]
            logger.info(
                f"Cached XBRL company facts for {ticker} are older than {self.cache_ttl_hours}h; refetching "
                f"(a new quarterly filing may have been published since the cache was written)."
            )

        logger.info(f"Fetching XBRL company facts for {ticker} (CIK {cik10})...")
        response = self._edgar._get(_COMPANYFACTS_URL_TEMPLATE.format(cik10=cik10))
        data = response.json()

        if not data.get("facts"):
            raise XbrlFactsUnavailableError(
                f"SEC returned no XBRL facts for {ticker} (CIK {cik10}). "
                f"This can happen for foreign private issuers filing under IFRS rather than US-GAAP."
            )

        payload = {"cached_at": datetime.now(timezone.utc).isoformat(), "data": data}
        with open(cache_path, "w") as f:
            json.dump(payload, f)
        return data

    # ---------- concept extraction ----------

    @staticmethod
    def _duration_days(entry: dict) -> Optional[int]:
        """Days between an XBRL fact's start/end. None for 'instant' facts (e.g. total assets) -- no ambiguity for these."""
        start, end = entry.get("start"), entry.get("end")
        if not start or not end:
            return None
        try:
            return (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days
        except ValueError:
            return None

    def _extract_concept(self, facts_json: dict, concept: str) -> Optional[dict]:
        """
        Returns the MOST RECENT value for a named concept, considering
        ALL known tag aliases together (must not stop at the first
        alias with any data -- a discontinued old tag can still have
        stale historical entries) and preferring the entry closest to
        one quarter in duration among those sharing the most recent
        end date (disambiguates quarterly vs. six-month figures).
        """
        candidates = self._all_candidates_for_concept(facts_json, concept)
        if not candidates:
            return None

        most_recent_end = max(c[2].get("end", "") for c in candidates)
        candidates_at_latest_end = [c for c in candidates if c[2].get("end", "") == most_recent_end]

        best = max(candidates_at_latest_end, key=self._quarter_closeness)
        return self._entry_to_result(concept, best)

    def _extract_concept_for_period(self, facts_json: dict, concept: str, target_period_end: str) -> Optional[dict]:
        """
        Like _extract_concept, but for a SPECIFIC target period-end
        date rather than always the globally most recent. This is what
        lets the evaluation harness build genuinely distinct historical
        scenarios instead of every "period" silently collapsing to
        whatever happens to be most recent at run time.
        """
        candidates = self._all_candidates_for_concept(facts_json, concept)
        candidates_at_target = [c for c in candidates if c[2].get("end", "") == target_period_end]
        if not candidates_at_target:
            return None

        best = max(candidates_at_target, key=self._quarter_closeness)
        return self._entry_to_result(concept, best)

    def _all_candidates_for_concept(self, facts_json: dict, concept: str) -> list[tuple[str, str, dict]]:
        us_gaap_facts = facts_json.get("facts", {}).get("us-gaap", {})
        candidates: list[tuple[str, str, dict]] = []
        for tag in _CONCEPT_TAG_ALIASES[concept]:
            tag_data = us_gaap_facts.get(tag)
            if not tag_data:
                continue
            for unit, entries in tag_data.get("units", {}).items():
                for entry in entries:
                    if entry.get("form") in _RELEVANT_FORMS:
                        candidates.append((tag, unit, entry))
        return candidates

    def _quarter_closeness(self, candidate: tuple[str, str, dict]) -> float:
        duration = self._duration_days(candidate[2])
        if duration is None:
            return 0.0  # instant-type fact: no ambiguity, always acceptable
        return -abs(duration - _TARGET_QUARTER_DAYS)

    @staticmethod
    def _entry_to_result(concept: str, candidate: tuple[str, str, dict]) -> dict:
        tag, unit, entry = candidate
        return {
            "concept": concept,
            "xbrl_tag": tag,
            "value": entry["val"],
            "unit": unit,
            "period_end": entry.get("end"),
            "period_start": entry.get("start"),
            "fiscal_year": entry.get("fy"),
            "fiscal_period": entry.get("fp"),
            "form": entry.get("form"),
            "filed_date": entry.get("filed"),
            "accession_number": entry.get("accn"),
        }

    # ---------- multi-period resolution (evaluation harness) ----------

    def get_available_quarterly_periods(self, ticker: str, max_periods: int = 3) -> list[dict]:
        """
        Returns metadata for the most recent `max_periods` DISTINCT
        quarterly reporting periods available for a ticker, resolved
        from the 'revenue' concept, sorted most recent first. Each
        entry: {"period_end", "period_start", "fiscal_year",
        "fiscal_period", "accession_number", "form", "calendar_quarter_label"}.

        Only entries with a genuine single-quarter duration
        (~91 +/- 15 days) are considered.

        Recency safeguard (fixed after a real bug found during
        development): a period whose end date is more than
        _MAX_PERIOD_AGE_DAYS old is excluded and logged as a warning,
        rather than silently returned as if it were current. This was
        found necessary for tickers (e.g. bank holding companies like
        JPM) whose income statement structure doesn't use a "Revenues"
        concept in recent filings -- without this safeguard, period
        resolution could silently fall back to the last year the
        company ever used a matching tag, potentially a decade stale,
        with no indication anything was wrong.

        calendar_quarter_label is derived directly from period_end
        (e.g. "2026-Q2"), not from XBRL's own fy/fp attributes -- those
        describe the filing's OWN context and can be inherited by
        comparative prior-year data points embedded in the same filing,
        which was found to produce misleading duplicate labels for two
        genuinely different historical quarters.
        """
        facts_json = self.get_company_facts(ticker)
        candidates = self._all_candidates_for_concept(facts_json, "revenue")

        quarterly_entries = [
            entry
            for _, _, entry in candidates
            if (d := self._duration_days(entry)) is not None
            and abs(d - _TARGET_QUARTER_DAYS) <= _QUARTER_DURATION_TOLERANCE_DAYS
        ]

        best_by_end: dict[str, dict] = {}
        for entry in quarterly_entries:
            end = entry.get("end", "")
            if end not in best_by_end or entry.get("filed", "") > best_by_end[end].get("filed", ""):
                best_by_end[end] = entry

        sorted_periods = sorted(best_by_end.values(), key=lambda e: e.get("end", ""), reverse=True)

        results = []
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=_MAX_PERIOD_AGE_DAYS)
        for e in sorted_periods:
            end_str = e.get("end", "")
            try:
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if end_date < cutoff:
                logger.warning(
                    f"Discarding stale resolved period for {ticker}: period_end={end_str} is older than "
                    f"{_MAX_PERIOD_AGE_DAYS} days. This usually means the filer doesn't use a 'Revenues'-family "
                    f"XBRL tag in its recent filings (common for banks/financial institutions) -- consider adding "
                    f"the correct tag alias for this filer's sector rather than trusting this stale fallback."
                )
                continue

            results.append(
                {
                    "period_end": end_str,
                    "period_start": e.get("start"),
                    "fiscal_year": e.get("fy"),
                    "fiscal_period": e.get("fp"),
                    "accession_number": e.get("accn"),
                    "form": e.get("form"),
                    "calendar_quarter_label": f"{end_date.year}-Q{(end_date.month - 1) // 3 + 1}",
                }
            )
            if len(results) >= max_periods:
                break

        return results

    def get_fundamentals_for_period(self, ticker: str, target_period_end: str) -> dict:
        """
        Returns fundamentals for a SPECIFIC historical quarter
        (identified by its period-end date, as returned by
        get_available_quarterly_periods()), rather than always the
        most recent. Used by the evaluation harness to build genuinely
        distinct scenarios across multiple quarters per ticker.
        """
        facts_json = self.get_company_facts(ticker)
        result = {"ticker": ticker.upper(), "period_end": target_period_end}
        for concept in _CONCEPT_TAG_ALIASES:
            result[concept] = self._extract_concept_for_period(facts_json, concept, target_period_end)
        return result

    def get_latest_fundamentals(self, ticker: str) -> dict:
        """
        Returns the most recent value of each key fundamental concept
        for a ticker, each carrying its own accession number. Used by
        FundamentalAgent for one-off, "current" analysis runs.
        """
        facts_json = self.get_company_facts(ticker)
        result = {"ticker": ticker.upper()}
        for concept in _CONCEPT_TAG_ALIASES:
            result[concept] = self._extract_concept(facts_json, concept)
        return result
