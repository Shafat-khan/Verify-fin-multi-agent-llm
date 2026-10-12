"""
verifyfin.agents.sentiment_agent
===================================
Sentiment Agent (Proposal Objective 2.3): produces a source-cited
ClaimObject characterising current market sentiment toward a ticker,
based on recent news headlines.

Architecturally different from FundamentalAgent/MacroAgent (important
distinction for the dissertation's Methodology chapter -- Literature
Review 2.6.3 discusses exactly this: different claim types need
different verification strategies):

    Fundamental/Macro claims are DETERMINISTIC facts the LLM only
    PHRASES -- the number is correct before the LLM ever sees it.
    Sentiment is not like that. "Is recent coverage of this company
    positive or negative" is a genuine INTERPRETIVE judgment; there is
    no ground-truth number to source deterministically. The LLM must
    actually read and judge the headlines.

    Because of this, confidence_score here reflects the LLM's own
    self-reported certainty in its interpretation (not a fixed high
    value like 0.95 used for directly-sourced XBRL/FRED facts), and
    the claim's SourceReference cites the SPECIFIC headline(s) that
    most informed the judgment -- so the claim is still auditable even
    though it isn't a bare fact lookup.

Known scope limitation (documented, not silently omitted): the
Verification Agent's knowledge base is built from SEC filing text
only. News headlines are not indexed there. Under the current
architecture, a sentiment claim's grounding_score against that
knowledge base is not meaningful, and such claims will generally route
to BLOCKED or CHALLENGED for reasons unrelated to whether the
sentiment judgment itself was reasonable. This is an intentional scope
boundary for MSc dissertation work, not an oversight -- extending the
knowledge base to index news content (or adding a dedicated
sentiment-verification path) is noted as future work rather than
attempted here. The claim's own SourceReference is still fully
traceable to a real headline regardless.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from verifyfin.data_pipeline.news_client import NewsClient, NewsUnavailableError
from verifyfin.llm.groq_client import GroqClient
from verifyfin.schemas.claim import AgentRole, ClaimCategory, ClaimObject, SourceReference
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise financial sentiment analyst. You will be given a list of recent "
    "news headlines about a company, each with an index number. Your job is to judge the "
    "OVERALL sentiment these headlines convey about the company.\n"
    "Rules:\n"
    "1. Base your judgment ONLY on the headlines given -- do not use any outside knowledge "
    "about the company.\n"
    "2. sentiment must be exactly one of: 'positive', 'negative', 'neutral'.\n"
    "3. supporting_headline_indices must list the index numbers of the 1-3 headlines that "
    "most directly informed your judgment.\n"
    "4. confidence must be a number between 0.0 and 1.0, reflecting how clear-cut the "
    "sentiment signal is -- use a LOWER value if headlines are mixed, ambiguous, or mostly "
    "unrelated to sentiment (e.g. routine product announcements).\n"
    "5. Reply with ONLY a JSON object, no other text, in this exact format:\n"
    '{"sentiment": "positive"|"negative"|"neutral", "claim_text": "<one sentence>", '
    '"supporting_headline_indices": [<int>, ...], "confidence": <float>}'
)

_DEFAULT_FALLBACK_CONFIDENCE = 0.4  # low: an unparseable response gives us no real signal about certainty


class SentimentAgent:
    def __init__(self, config_path: Optional[str] = None, llm_client: Optional[GroqClient] = None):
        self.config_path = config_path
        self._news = NewsClient(config_path)
        self._llm = llm_client if llm_client is not None else GroqClient(config_path)

    @staticmethod
    def _parse_sentiment_response(raw_response: str) -> Optional[dict]:
        """
        Parses the LLM's JSON sentiment judgment. Tolerant of markdown
        code fences. Returns None if unparseable or missing required
        fields -- there is no safe deterministic fallback for an
        interpretive judgment (unlike FundamentalAgent's numeric
        fallback template), so an unparseable response means no claim
        is produced for this run rather than guessing.
        """
        text = raw_response.strip()
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Sentiment Agent response as JSON ({e}).")
            return None

        if not isinstance(parsed, dict):
            return None
        if parsed.get("sentiment") not in ("positive", "negative", "neutral"):
            logger.warning(f"Sentiment Agent returned an unrecognised sentiment value: {parsed.get('sentiment')!r}")
            return None
        if "claim_text" not in parsed:
            return None

        confidence = parsed.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            logger.warning(f"Sentiment Agent returned an invalid confidence value: {confidence!r}; using fallback.")
            parsed["confidence"] = _DEFAULT_FALLBACK_CONFIDENCE

        if not isinstance(parsed.get("supporting_headline_indices"), list):
            parsed["supporting_headline_indices"] = []

        return parsed

    def _build_sentiment_claim(
        self, ticker: str, time_period: str, articles: list[dict]
    ) -> Optional[ClaimObject]:
        payload_for_llm = [
            {"index": i, "title": a["title"], "published": a["published"]} for i, a in enumerate(articles)
        ]
        user_prompt = f"Recent headlines about {ticker}:\n{json.dumps(payload_for_llm, indent=2)}"

        raw_response = self._llm.complete(user_prompt=user_prompt, system_prompt=_SYSTEM_PROMPT)
        parsed = self._parse_sentiment_response(raw_response)

        if parsed is None:
            logger.warning(f"No usable sentiment judgment produced for {ticker}; skipping sentiment claim.")
            return None

        supporting_indices = [i for i in parsed["supporting_headline_indices"] if 0 <= i < len(articles)]
        supporting_articles = [articles[i] for i in supporting_indices] or articles[:1]
        excerpt = "; ".join(f'"{a["title"]}" ({a["source"] or "unknown source"})' for a in supporting_articles)

        return ClaimObject(
            claim_text=parsed["claim_text"],
            claim_category=ClaimCategory.SENTIMENT,
            produced_by=AgentRole.SENTIMENT,
            source_reference=SourceReference(
                source_type="news",
                document_id=supporting_articles[0]["link"] or ticker.upper(),
                excerpt=excerpt,
            ),
            confidence_score=float(parsed["confidence"]),
            ticker=ticker.upper(),
            time_period=time_period,
        )

    def analyze(self, ticker: str, time_period: str, company_name: Optional[str] = None) -> list[ClaimObject]:
        """
        Produces a single sentiment ClaimObject for the ticker, based on
        recent news headlines. Returns an empty list (not an error) if
        no news is available or the LLM's judgment couldn't be parsed
        -- a missing sentiment claim degrades the report gracefully
        rather than blocking the whole pipeline.
        """
        logger.info(f"Sentiment Agent analyzing {ticker}...")
        try:
            articles = self._news.get_recent_headlines(ticker, company_name=company_name)
        except NewsUnavailableError as e:
            logger.warning(f"No news available for {ticker}: {e}")
            return []

        claim = self._build_sentiment_claim(ticker, time_period, articles)
        if claim is None:
            return []

        logger.info(f"Sentiment Agent produced 1 claim for {ticker}.")
        return [claim]
