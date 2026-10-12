"""
verifyfin.agents.verification_agent
======================================
Verification Agent (Proposal Objective 2.4; Literature Review Gap 3):
the core novelty of VERIFY-FIN. Takes claims produced by analyst
agents and checks each one against the vector store knowledge base,
computing a grounding score and routing each claim to
GROUNDED / CHALLENGED / BLOCKED via ClaimObject.mark_grounded().

Hybrid semantic + numeric verification: a wide candidate pool is
retrieved via cosine similarity, then any candidate that literally
contains the claim's source number is preferred over the top
semantic-only match.

Date-aware numeric matching (added after a real ambiguity found during
the first full evaluation run): a value alone is not always enough to
uniquely identify the correct evidence -- e.g. a monthly unemployment
rate can stay unchanged for several consecutive months, so matching on
the number alone can find a real, correctly-formatted observation that
nonetheless belongs to the WRONG month. This was caught by the Debate
Agent ("the evidence reports the rate for 2026-08-01, not the claimed
date of 2026-07-01"), which is exactly the kind of subtlety a second,
independent check should catch -- but it's better caught earlier, by
preferring a candidate that matches BOTH the value and the claim's
stated date when one is available, falling back to value-only matching
when no such combined match exists.

Excerpt formats vary by source type: FRED excerpts have the form
"SERIES: value (as of date)" with no unit token, unlike XBRL's
"TAG: value UNIT (period ...)" -- the unit group in the parsing regex
is optional to handle both.
"""

from __future__ import annotations

import re
from typing import Optional

from verifyfin.agents import evidence_alignment as ea
from verifyfin.agents.text_consistency import check_text_against_excerpt
from verifyfin.data_pipeline.vector_store import VectorStore
from verifyfin.schemas.claim import ClaimObject, GroundingVerdict
from verifyfin.utils.io_utils import load_config
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Matches "TAG: value UNIT ..." (XBRL) or "SERIES: value (as of ...)" (FRED)
# -- the unit group is optional since FRED excerpts have no unit token.
_EXCERPT_VALUE_PATTERN = re.compile(r":\s*(-?[\d.]+)(?:\s+([A-Za-z/]+))?")

# Matches an ISO date (YYYY-MM-DD) anywhere in an excerpt string.
_EXCERPT_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")

_NUMERIC_MATCH_BOOST = 0.25  # added to similarity when the claim's exact source number is found verbatim


class VerificationAgent:
    def __init__(
        self,
        config_path: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        *,
        use_numeric_match: bool = True,
        use_targeted_search: bool = True,
        use_date_preference: bool = True,
        require_date_match: bool = False,
        numeric_boost: float = _NUMERIC_MATCH_BOOST,
        check_text_consistency: bool = False,
        text_mismatch_action: str = "challenge",
        date_aligned_search: bool = False,
        label_check: bool = False,
        filing_period_check: bool = False,
        exact_record_grounds: bool = False,
    ):
        """
        Ablation switches (keyword-only; the defaults reproduce the production
        behaviour exactly, so existing results are unaffected):

          use_numeric_match   False -> pure cosine similarity (the proposal's original design)
          use_targeted_search False -> no second-stage `$contains` search for the number
          use_date_preference False -> match on the value alone, ignoring the claim's date
          require_date_match  True  -> a FRED observation chunk counts as a numeric match
                                       only if it ALSO contains the claim's date (stricter than
                                       "prefer"). Filing chunks use month-name dates, so they
                                       are never date-constrained.
          numeric_boost       size of the score boost when the exact number is found (0 = none)
          check_text_consistency
                              True -> ALSO require that the number and date written in the claim
                                      TEXT agree with the source excerpt (see text_consistency.py).
                                      Added after fault injection showed the verifier caught only
                                      ~13% of wrong sentences whose cited excerpt was correct.
                                      Default False so earlier results stay reproducible.
          text_mismatch_action
                              what a text/excerpt mismatch does: "challenge" caps the score just
                              below the GROUNDED threshold (the Debate Agent then decides);
                              "block" caps it below the CHALLENGED threshold (claim is dropped).

          v3 evidence-alignment switches (all default False; see evidence_alignment.py for the why):
          date_aligned_search  FRED claims: look up the EXACT record "SERIES observation: value on DATE" with a
                               substring search on the date, instead of trusting a value match inside the top-8
                               pool. Found -> that chunk is the evidence (this repairs a wrong-date citation).
                               Not found -> the claim cannot be GROUNDED (capped to CHALLENGED).
          label_check          the series (FRED) or line item (filings) the SENTENCE names must match the matched
                               chunk; a positive mismatch caps the score to CHALLENGED.
          filing_period_check  filing claims: the table column holding the number must carry the claimed
                               period-end date; on a mismatch, try another candidate chunk, else cap to CHALLENGED.
          exact_record_grounds a claim confirmed as an exact structured record (FRED: exact record + series ok;
                               filings: line item AND period column both confirmed) is routed to GROUNDED even if
                               cosine similarity is low (an exact lookup is stronger evidence than cosine).
                               Text-consistency and mismatch caps still win.
        """
        if text_mismatch_action not in ("challenge", "block"):
            raise ValueError("text_mismatch_action must be 'challenge' or 'block'")
        self.check_text_consistency = check_text_consistency
        self.text_mismatch_action = text_mismatch_action
        self.text_check_counts = {"ok": 0, "mismatch": 0, "not_stated": 0}
        self.date_aligned_search = date_aligned_search
        self.label_check = label_check
        self.filing_period_check = filing_period_check
        self.exact_record_grounds = exact_record_grounds
        self.alignment_counts = {"record_found": 0, "record_missing": 0, "label_mismatch": 0, "period_mismatch": 0,
                                 "period_reselected": 0, "exact_lift": 0}
        self.use_numeric_match = use_numeric_match
        self.use_targeted_search = use_targeted_search
        self.use_date_preference = use_date_preference
        self.require_date_match = require_date_match
        self.numeric_boost = numeric_boost
        self.config = load_config(config_path)
        grounding_cfg = self.config["grounding"]
        self.grounded_threshold = grounding_cfg["grounded_threshold"]
        self.challenged_threshold = grounding_cfg["challenged_threshold"]

        self._store = vector_store if vector_store is not None else VectorStore(config_path)

    def _query_text_for_claim(self, claim: ClaimObject) -> str:
        """Always claim_text (natural language) -- see module docstring for why."""
        return claim.claim_text

    @staticmethod
    def _extract_value_and_unit_from_excerpt(excerpt: Optional[str]) -> tuple[Optional[float], Optional[str]]:
        if not excerpt:
            return None, None
        match = _EXCERPT_VALUE_PATTERN.search(excerpt)
        if not match:
            return None, None
        try:
            return float(match.group(1)), match.group(2)
        except ValueError:
            return None, None

    @staticmethod
    def _extract_date_from_excerpt(excerpt: Optional[str]) -> Optional[str]:
        """
        Extracts an ISO date (YYYY-MM-DD) from an excerpt, if present.
        If multiple dates appear (e.g. "period 2025-12-29 to 2026-03-28"),
        the LAST one is used, since that's consistently the period-end /
        "as of" date the claim's value actually describes.
        """
        if not excerpt:
            return None
        matches = _EXCERPT_DATE_PATTERN.findall(excerpt)
        return matches[-1] if matches else None

    @staticmethod
    def _number_text_variants(value: Optional[float], unit: Optional[str]) -> set[str]:
        variants: set[str] = set()
        if value is None:
            return variants

        if value == int(value):
            variants.add(f"{int(value):,}")
            variants.add(str(int(value)))
        else:
            variants.add(f"{value:,}")
            variants.add(str(value))

        if unit == "USD" and abs(value) >= 1_000_000:
            millions = value / 1_000_000
            if millions == int(millions):
                variants.add(f"{int(millions):,}")
                variants.add(str(int(millions)))
            else:
                variants.add(f"{millions:,.1f}")

        return variants

    @staticmethod
    def _contains_number_variant(text: str, variants: set[str]) -> bool:
        return any(v in text for v in variants)

    def verify_claim(self, claim: ClaimObject, top_k: int = 8) -> ClaimObject:
        """
        Verifies a single claim IN PLACE (mutates and returns it).

        1. Retrieve a candidate pool via unrestricted semantic search.
        2. Among candidates already containing the claim's exact source
           number, prefer one that ALSO contains the claim's stated
           date (if any) -- disambiguates repeated values across
           periods (see module docstring). Falls back to a value-only
           match if no date-qualified match exists among the initial
           candidates.
        3. If no numeric match exists in the initial pool at all, run a
           TARGETED search restricted to chunks containing the number,
           again preferring a date-qualified result.

        An empty knowledge base routes to BLOCKED, not an error.
        """
        query_text = self._query_text_for_claim(claim)
        results = self._store.query(query_text, ticker=claim.ticker, top_k=top_k)

        if not results:
            logger.warning(
                f"No evidence found in the knowledge base for claim "
                f"'{claim.claim_text[:60]}...' (ticker={claim.ticker}). Routing to BLOCKED."
            )
            claim.mark_grounded(
                grounding_score=0.0,
                evidence="(no evidence retrieved from knowledge base)",
                threshold_grounded=self.grounded_threshold,
                threshold_challenged=self.challenged_threshold,
            )
            return claim

        value, unit = self._extract_value_and_unit_from_excerpt(claim.source_reference.excerpt)
        variants = self._number_text_variants(value, unit)
        claim_date = (self._extract_date_from_excerpt(claim.source_reference.excerpt)
                      if self.use_date_preference else None)

        numeric_match = None
        method = "semantic_only"

        if variants and self.use_numeric_match:
            value_matches = [r for r in results if self._contains_number_variant(r["text"], variants)]
            numeric_match = self._prefer_date_qualified(value_matches, claim_date, require=self.require_date_match)

            if numeric_match is not None:
                method = "semantic+numeric_match"
            else:
                if self.use_targeted_search:
                    numeric_match = self._targeted_numeric_search(
                        query_text, claim.ticker, variants, claim_date, require=self.require_date_match
                    )
                if numeric_match is not None:
                    method = "targeted_numeric_search"

        if numeric_match is not None:
            score = min(1.0, numeric_match["similarity_score"] + self.numeric_boost)
            evidence = numeric_match["text"]
        else:
            score = results[0]["similarity_score"]
            evidence = results[0]["text"]

        if self.date_aligned_search or self.label_check or self.filing_period_check or self.exact_record_grounds:
            score, evidence, method = self._apply_alignment(
                claim, query_text, results, value, variants, self._extract_date_from_excerpt(claim.source_reference.excerpt),
                numeric_match, score, evidence, method)

        if self.check_text_consistency:
            check = check_text_against_excerpt(
                claim.claim_text, value, self._extract_date_from_excerpt(claim.source_reference.excerpt))
            self.text_check_counts["mismatch" if not check.consistent else
                                   ("ok" if "ok" in (check.number, check.date) else "not_stated")] += 1
            if not check.consistent:
                cap = (self.grounded_threshold if self.text_mismatch_action == "challenge"
                       else self.challenged_threshold) - 0.001
                score = min(score, cap)
                method += f"+text_mismatch({check.detail})"

        claim.mark_grounded(
            grounding_score=score,
            evidence=evidence,
            threshold_grounded=self.grounded_threshold,
            threshold_challenged=self.challenged_threshold,
        )

        verdict_str = claim.verdict.value if hasattr(claim.verdict, "value") else claim.verdict
        logger.info(
            f"Verified claim (score={claim.grounding_score:.3f}, verdict={verdict_str}, method={method}): "
            f"'{claim.claim_text[:60]}...'"
        )
        return claim

    def _apply_alignment(self, claim, query_text, results, value, variants, excerpt_date, numeric_match, score, evidence, method):
        """
        v3 evidence-alignment (see evidence_alignment.py). Runs after the v1 matching step and BEFORE the v2 text
        check; every cap uses min(), so the order of the caps does not matter. Returns (score, evidence, method).
        """
        if not variants:
            return score, evidence, method
        counts = self.alignment_counts
        cap = self.grounded_threshold - 0.001
        confirmed = False      # the matched passage is positively confirmed as the exact structured record
        mismatch = False
        sentence = claim.claim_text

        if claim.source_reference.source_type == "fred":
            record = None
            if self.date_aligned_search and excerpt_date:
                candidates = self._store.query(query_text, ticker=claim.ticker, top_k=20, where_document_contains=excerpt_date)
                record = ea.pick_record(ea.fred_records(candidates, variants, excerpt_date), ea.claim_series(sentence))
                if record is not None:
                    counts["record_found"] += 1
                    numeric_match, evidence = record, record["text"]
                    score = min(1.0, record["similarity_score"] + self.numeric_boost)
                    method += "+fred_record"
                else:
                    counts["record_missing"] += 1
                    score = min(score, cap)
                    method += "+fred_record_missing"
            series_ok = False
            if self.label_check and numeric_match is not None:
                status = ea.series_status(sentence, evidence)
                if status == "mismatch":
                    counts["label_mismatch"] += 1
                    mismatch = True
                    method += "+label_mismatch"
                series_ok = status == "ok"
            confirmed = record is not None and series_ok
        elif numeric_match is not None:
            label = ea.label_status(sentence, evidence) if self.label_check else "unknown"
            period = ea.period_status(evidence, variants, excerpt_date) if self.filing_period_check else "unknown"
            if period == "mismatch":
                alternative = self._period_aligned_candidate(query_text, claim, results, variants, excerpt_date)
                if alternative is not None:
                    counts["period_reselected"] += 1
                    numeric_match, evidence = alternative, alternative["text"]
                    score = min(1.0, alternative["similarity_score"] + self.numeric_boost)
                    method += "+period_reselected"
                    period = "ok"
                    label = ea.label_status(sentence, evidence) if self.label_check else "unknown"
            if label == "mismatch":
                counts["label_mismatch"] += 1
                mismatch = True
                method += "+label_mismatch"
            if period == "mismatch":
                counts["period_mismatch"] += 1
                mismatch = True
                method += "+period_column_mismatch"
            confirmed = label == "ok" and period == "ok"

        if mismatch:
            score = min(score, cap)
        elif self.exact_record_grounds and confirmed and score < self.grounded_threshold + 0.001:
            score = self.grounded_threshold + 0.001
            counts["exact_lift"] += 1
            method += "+exact_record"
        return score, evidence, method

    def _period_aligned_candidate(self, query_text, claim, results, variants, excerpt_date) -> Optional[dict]:
        """Another chunk that holds the claimed number in the column of the claimed period (and, if enabled, the claimed line item)."""
        pool = [r for r in results if ea.contains_number(r["text"], variants)]
        for variant in sorted(variants, key=len, reverse=True)[:2]:
            pool += self._store.query(query_text, ticker=claim.ticker, top_k=10, where_document_contains=variant)
        for r in pool:
            if ea.period_status(r["text"], variants, excerpt_date) != "ok":
                continue
            if self.label_check and ea.label_status(claim.claim_text, r["text"]) == "mismatch":
                continue
            return r
        return None

    @staticmethod
    def _prefer_date_qualified(
        candidates: list[dict], claim_date: Optional[str], require: bool = False
    ) -> Optional[dict]:
        """
        Given candidates that already match the claim's number, prefer one that ALSO contains the
        claim's stated date. Default (require=False): fall back to the first candidate if none
        carries the date (value-only match). Strict (require=True): a FRED observation chunk is
        accepted only if it contains the date; chunks from other sources (filings use month-name
        dates, not ISO) are not date-constrained. Returns None if nothing qualifies.
        """
        if not candidates:
            return None
        if claim_date:
            date_matched = [c for c in candidates if claim_date in c["text"]]
            if date_matched:
                return date_matched[0]
            if require:
                non_fred = [c for c in candidates if (c.get("metadata") or {}).get("source_type") != "fred"]
                return non_fred[0] if non_fred else None
        return candidates[0]

    def _targeted_numeric_search(
        self,
        query_text: str,
        ticker: str,
        variants: set[str],
        claim_date: Optional[str] = None,
        require: bool = False,
    ) -> Optional[dict]:
        """
        Searches for chunks containing one of the claim's number variants using ChromaDB's document
        `$contains` filter, longest variant first, preferring (or, if require, demanding for FRED
        chunks) a chunk that also contains claim_date.
        """
        for variant in sorted(variants, key=len, reverse=True):
            results = self._store.query(query_text, ticker=ticker, top_k=10, where_document_contains=variant)
            match = self._prefer_date_qualified(results, claim_date, require=require)
            if match is not None:
                return match
        return None

    def verify_claims(self, claims: list[ClaimObject], top_k: int = 8) -> list[ClaimObject]:
        logger.info(f"Verification Agent checking {len(claims)} claims...")

        for claim in claims:
            self.verify_claim(claim, top_k=top_k)

        grounded = sum(1 for c in claims if c.verdict == GroundingVerdict.GROUNDED)
        challenged = sum(1 for c in claims if c.verdict == GroundingVerdict.CHALLENGED)
        blocked = sum(1 for c in claims if c.verdict == GroundingVerdict.BLOCKED)
        logger.info(f"Verification complete: {grounded} GROUNDED, {challenged} CHALLENGED, {blocked} BLOCKED.")

        return claims
