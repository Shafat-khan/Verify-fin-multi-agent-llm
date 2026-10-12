"""
verifyfin.data_pipeline.edgar_client
=======================================
Client for retrieving 10-K / 10-Q filings from SEC EDGAR for the
Fundamental Agent's knowledge base (Proposal Objective 2.3).

Three-step retrieval process, matching how EDGAR is actually structured:
    1. Ticker -> CIK (Central Index Key) lookup, via SEC's static
       company_tickers.json mapping file.
    2. CIK -> list of recent filings, via data.sec.gov's submissions API.
    3. Filing -> raw document -> cleaned plain text, with a fallback
       parser for malformed markup (Proposal §4, "Potential Issues":
       "EDGAR scraping may occasionally return malformed XML. A robust
       parser with fallback handling will be implemented.")

SEC EDGAR access rules this client respects:
    - A descriptive User-Agent header identifying the requester is
      required by SEC; requests without one are rejected with 403/404.
    - SEC asks that automated tools stay under ~10 requests/second;
      this client adds a small politeness delay between requests.
    - All responses are cached to disk so re-running the pipeline never
      re-fetches a filing it already has.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from lxml import html as lxml_html

from verifyfin.utils.io_utils import load_config, resolve_path
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik10}.json"
_ARCHIVES_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"

_MIN_REQUEST_INTERVAL_SECONDS = 0.15  # politeness delay, keeps us well under SEC's ~10 req/s guidance


class EdgarFilingNotFoundError(Exception):
    """Raised when no filing of the requested type(s) can be found for a ticker."""


class EdgarClient:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        edgar_cfg = self.config["data_sources"]["edgar"]

        user_agent = edgar_cfg.get("user_agent", "")
        if "replace_with_university_email" in user_agent:
            logger.warning(
                "configs/config.yaml still has the placeholder EDGAR user_agent contact email. "
                "SEC requests a real contact identifier -- update data_sources.edgar.user_agent "
                "before running large jobs."
            )
        self._headers = {"User-Agent": user_agent}

        self.filing_types = edgar_cfg.get("filing_types", ["10-K", "10-Q"])
        self.cache_dir = resolve_path("data/raw/edgar_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_time = 0.0

    # ---------- low-level HTTP with politeness delay + retry ----------

    def _get(self, url: str, max_retries: int = 3) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)

        last_exception: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, headers=self._headers, timeout=15)
                self._last_request_time = time.monotonic()
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_exception = e
                logger.warning(f"EDGAR request failed (attempt {attempt}/{max_retries}) for {url}: {e}")
                time.sleep(0.5 * attempt)  # simple linear backoff

        raise RuntimeError(f"EDGAR request failed after {max_retries} attempts: {url}") from last_exception

    # ---------- step 1: ticker -> CIK ----------

    def _load_ticker_map(self) -> dict:
        cache_path = self.cache_dir / "company_tickers.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)

        logger.info("Fetching SEC ticker-to-CIK mapping (first run only, will be cached)...")
        response = self._get(_TICKER_MAP_URL)
        data = response.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data

    def get_cik_for_ticker(self, ticker: str) -> str:
        """Returns the 10-digit, zero-padded CIK string for a ticker, e.g. 'AAPL' -> '0000320193'."""
        ticker_map = self._load_ticker_map()
        ticker_upper = ticker.upper().strip()

        for entry in ticker_map.values():
            if entry["ticker"].upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)

        raise ValueError(f"Ticker '{ticker}' not found in SEC's company_tickers.json mapping.")

    # ---------- step 2: CIK -> filing list ----------

    def get_filings_list(
        self, ticker: str, filing_types: Optional[list[str]] = None, max_filings: int = 4
    ) -> list[dict]:
        """
        Returns metadata for the most recent filings of the requested
        type(s), most recent first. Each entry contains: form,
        accession_number, filing_date, report_date, primary_document, cik.
        """
        filing_types = filing_types or self.filing_types
        cik10 = self.get_cik_for_ticker(ticker)

        response = self._get(_SUBMISSIONS_URL_TEMPLATE.format(cik10=cik10))
        data = response.json()
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_documents = recent.get("primaryDocument", [])

        matches = []
        for i, form in enumerate(forms):
            if form in filing_types:
                matches.append(
                    {
                        "ticker": ticker.upper(),
                        "cik": cik10,
                        "form": form,
                        "accession_number": accession_numbers[i],
                        "filing_date": filing_dates[i],
                        "report_date": report_dates[i] if i < len(report_dates) else None,
                        "primary_document": primary_documents[i],
                    }
                )
                if len(matches) >= max_filings:
                    break

        if not matches:
            raise EdgarFilingNotFoundError(
                f"No filings of type {filing_types} found for {ticker} (CIK {cik10})."
            )

        return matches

    # ---------- step 3: filing -> raw document -> clean text ----------

    def download_filing(self, filing_meta: dict) -> str:
        """
        Downloads the raw document (HTML) for a filing, caching to disk.
        `filing_meta` is one entry as returned by get_filings_list().
        """
        accession_nodash = filing_meta["accession_number"].replace("-", "")
        cik_int = int(filing_meta["cik"])
        url = _ARCHIVES_URL_TEMPLATE.format(
            cik_int=cik_int, accession_nodash=accession_nodash, primary_doc=filing_meta["primary_document"]
        )

        cache_path = self.cache_dir / f"{filing_meta['ticker']}_{accession_nodash}.html"
        if cache_path.exists():
            logger.debug(f"Using cached filing: {cache_path.name}")
            return cache_path.read_text(encoding="utf-8", errors="ignore")

        logger.info(f"Downloading {filing_meta['ticker']} {filing_meta['form']} ({filing_meta['filing_date']})...")
        response = self._get(url)
        raw_html = response.text
        cache_path.write_text(raw_html, encoding="utf-8")
        return raw_html

    def parse_filing_text(self, raw_html: str) -> str:
        """
        Extracts clean plain text from a filing document.

        Script/style blocks are stripped from the RAW markup first, via
        regex, before lxml ever sees it (see class docstring for why:
        malformed adjacent tags can trick lxml into not recognizing
        script/style as real elements).

        TABLE HANDLING (important for a project grounding numeric
        claims): SEC filings' financial statements are HTML tables.
        lxml's default text_content() concatenates every cell's text
        with no separation, which silently destroys the association
        between a label (e.g. "Total net sales") and its value (e.g.
        "254,940") once they're mashed together with dozens of other
        numbers in the same table. This was found during development to
        materially suppress retrieval/grounding quality: claims
        generated directly from a filing's own figures were scoring far
        below the grounding threshold against that same filing's text,
        because the embedding model could no longer see which number
        belonged to which label. Tables are therefore extracted
        separately, row by row, with cells joined by " | " (a literal
        token that survives the chunker's word-based splitting, unlike
        whitespace) so label-value adjacency is preserved.

        Primary strategy: lxml's HTML parser (tolerant of most malformed
        markup via libxml2's recovery mode), with table-aware extraction
        layered on top.

        Fallback strategy (Proposal §4 risk mitigation): if lxml still
        fails or produces empty output on a sufficiently malformed
        document, fall back to a crude but robust regex-based tag
        stripper that cannot itself raise on malformed input. The
        fallback does not attempt table-awareness -- it is a last
        resort for documents too broken to parse structurally at all.
        """
        prescrubbed = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        prescrubbed = re.sub(r"<style.*?</style>", " ", prescrubbed, flags=re.DOTALL | re.IGNORECASE)

        try:
            tree = lxml_html.fromstring(prescrubbed.encode("utf-8", errors="ignore"))
            for bad in tree.xpath("//script | //style"):
                bad.getparent().remove(bad)

            text = self._extract_text_preserving_tables(tree)
            # Collapse repeated horizontal whitespace, but preserve the
            # newlines inserted between table rows for readability.
            cleaned = re.sub(r"[ \t]+", " ", text)
            cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
            if cleaned:
                return cleaned
            raise ValueError("lxml parsed the document but extracted no text content.")
        except Exception as e:
            logger.warning(f"Primary lxml parsing failed or produced empty text ({e}); using regex fallback parser.")
            text = re.sub(r"<[^>]+>", " ", prescrubbed)
            text = re.sub(r"&nbsp;|&amp;|&#160;", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

    _NUMERIC_CELL_PATTERN = re.compile(r"^\(?-?\$?\d{1,3}(,\d{3})*(\.\d+)?\)?%?$")

    @classmethod
    def _looks_like_data_row(cls, cells: list[str]) -> bool:
        """
        A row is treated as a data row if any of its cells looks like a
        numeric value: an integer, decimal, comma-grouped figure, or
        percentage, optionally with a leading $ or parentheses for a
        negative number (e.g. "254,940", "4.87", "(26,677)", "12.5%").
        Rows before the first such row in a table are treated as header
        context. Plain dates ("March 28, 2026") and labels do not match
        this pattern, since the whole cell must be numeric.
        """
        return any(cls._NUMERIC_CELL_PATTERN.match(cell.replace(" ", "")) for cell in cells)

    @classmethod
    def _extract_text_preserving_tables(cls, tree) -> str:
        """
        Extracts all table content, row by row, with cells joined by
        " | " to preserve label-value adjacency, then removes the
        tables from the tree and appends the remaining (non-table) prose
        text_content().

        Multi-row header context propagation (fixed after a real bug
        found during development -- caught by VERIFY-FIN's own Debate
        Agent). An earlier version of this method only treated a
        table's FIRST row as its header. Many SEC financial-statement
        tables actually use a TWO-row header: a group-label row (e.g.
        "Three Months Ended | Six Months Ended") followed by a row with
        the actual dates ("March 28, 2026 | March 29, 2025 | ..."). If
        only the first row is captured, every data row is prepended
        with an INCOMPLETE header that names the period grouping but
        not the specific dates -- which was found to cause the Debate
        Agent to correctly, rigorously reject a claim ("net income")
        because the evidence "does not specify the quarter", since the
        actual dates were never attached to that row's chunk.

        This version instead accumulates ALL leading rows that do not
        look like a data row (see _looks_like_data_row) as the combined
        header, so a two-row (or more) header is captured in full.

        Note on ordering: this appends prose AFTER all table content,
        rather than interleaving tables in their original document
        position -- a documented simplification appropriate for MSc
        dissertation scope.
        """
        table_parts: list[str] = []
        for table in tree.xpath("//table"):
            rows_cells: list[list[str]] = []
            for row in table.xpath(".//tr"):
                cells = [c.text_content().strip() for c in row.xpath("./td | ./th")]
                cells = [c for c in cells if c]
                if cells:
                    rows_cells.append(cells)

            if rows_cells:
                header_row_texts: list[str] = []
                data_start_idx = len(rows_cells)  # default: no data row found at all
                for idx, cells in enumerate(rows_cells):
                    if cls._looks_like_data_row(cells):
                        data_start_idx = idx
                        break
                    header_row_texts.append(" | ".join(cells))

                combined_header = " | ".join(header_row_texts) if header_row_texts else None

                for header_row_text in header_row_texts:
                    table_parts.append(header_row_text)  # each header row also stands alone as its own chunk

                for cells in rows_cells[data_start_idx:]:
                    row_text = " | ".join(cells)
                    if combined_header:
                        table_parts.append(f"{combined_header} || {row_text}")
                    else:
                        table_parts.append(row_text)

            parent = table.getparent()
            if parent is not None:
                parent.remove(table)  # avoid double-counting table text in the prose pass below

        prose_text = tree.text_content()
        return "\n".join(table_parts) + "\n" + prose_text

    # ---------- convenience: full pipeline for one ticker ----------

    def get_filings_for_ticker(
        self, ticker: str, filing_types: Optional[list[str]] = None, max_filings: int = 4
    ) -> list[dict]:
        """
        Full pipeline: ticker -> filing list -> downloaded + cleaned text
        for each filing. Returns each filing_meta dict with an added
        'clean_text' key, ready for chunking (verifyfin.data_pipeline.chunker).
        """
        filings = self.get_filings_list(ticker, filing_types=filing_types, max_filings=max_filings)
        for filing in filings:
            raw_html = self.download_filing(filing)
            filing["clean_text"] = self.parse_filing_text(raw_html)
            logger.info(
                f"{ticker} {filing['form']} ({filing['filing_date']}): "
                f"extracted {len(filing['clean_text'])} characters of clean text."
            )
        return filings
