"""
Unit tests for verifyfin.schemas.claim.

These tests substantiate the core architectural claim made in the
dissertation (Lit. Review, Gap 2): it is structurally impossible to
construct a ClaimObject without a source reference and a confidence
score, unlike free-form-prose multi-agent systems.
"""

import pytest
from pydantic import ValidationError

from verifyfin.schemas.claim import (
    AgentRole,
    ClaimCategory,
    ClaimObject,
    GroundingVerdict,
    SourceReference,
)


def make_valid_claim(**overrides) -> ClaimObject:
    defaults = dict(
        claim_text="AAPL reported revenue of $94.9B in Q3 2024.",
        claim_category=ClaimCategory.NUMERICAL,
        produced_by=AgentRole.FUNDAMENTAL,
        source_reference=SourceReference(
            source_type="edgar_10q",
            document_id="0000320193-24-000081",
            excerpt="Total net sales of $94.9 billion",
        ),
        confidence_score=0.92,
        ticker="AAPL",
        time_period="2024-Q3",
    )
    defaults.update(overrides)
    return ClaimObject(**defaults)


class TestClaimObjectConstruction:
    def test_valid_claim_constructs_successfully(self):
        claim = make_valid_claim()
        assert claim.claim_text.startswith("AAPL reported revenue")
        assert claim.verdict == GroundingVerdict.UNVERIFIED
        assert claim.grounding_score is None

    def test_claim_without_source_reference_is_rejected(self):
        # This is the central architectural guarantee: no ClaimObject
        # can exist without a source_reference, unlike free-form prose
        # inter-agent communication in prior systems.
        with pytest.raises(ValidationError):
            ClaimObject(
                claim_text="AAPL revenue grew.",
                claim_category=ClaimCategory.NUMERICAL,
                produced_by=AgentRole.FUNDAMENTAL,
                confidence_score=0.9,
                ticker="AAPL",
                time_period="2024-Q3",
                # source_reference omitted deliberately
            )

    def test_claim_without_confidence_score_is_rejected(self):
        with pytest.raises(ValidationError):
            make_valid_claim(confidence_score=None)

    def test_confidence_score_out_of_range_is_rejected(self):
        with pytest.raises(ValidationError):
            make_valid_claim(confidence_score=1.5)

    @pytest.mark.parametrize("placeholder", ["", "N/A", "todo", "TBD", "   "])
    def test_placeholder_claim_text_is_rejected(self, placeholder):
        with pytest.raises(ValidationError):
            make_valid_claim(claim_text=placeholder)


class TestGroundingRouting:
    def test_high_score_routes_to_grounded(self):
        claim = make_valid_claim()
        claim.mark_grounded(
            grounding_score=0.85,
            evidence="Total net sales of $94.9 billion in Q3 2024.",
            threshold_grounded=0.75,
            threshold_challenged=0.55,
        )
        assert claim.verdict == GroundingVerdict.GROUNDED
        assert claim.is_usable_in_final_report() is True

    def test_mid_score_routes_to_challenged(self):
        claim = make_valid_claim()
        claim.mark_grounded(
            grounding_score=0.6,
            evidence="Some loosely related passage.",
            threshold_grounded=0.75,
            threshold_challenged=0.55,
        )
        assert claim.verdict == GroundingVerdict.CHALLENGED
        # Not usable until the Debate Agent upholds it
        assert claim.is_usable_in_final_report() is False

        claim.debate_resolution = "upheld"
        assert claim.is_usable_in_final_report() is True

    def test_low_score_routes_to_blocked(self):
        claim = make_valid_claim()
        claim.mark_grounded(
            grounding_score=0.2,
            evidence="Unrelated passage.",
            threshold_grounded=0.75,
            threshold_challenged=0.55,
        )
        assert claim.verdict == GroundingVerdict.BLOCKED
        assert claim.is_usable_in_final_report() is False

    def test_negation_case_documented_limitation(self):
        """
        Documents the known weakness (Lit. Review 2.4.3): cosine
        similarity measures relatedness, not entailment, so a claim and
        its negation can both score highly. This test does not assert
        that the system detects the negation (it currently cannot) --
        it documents the behaviour so it is visible in the test suite
        and referenced in the dissertation's Limitations section.
        """
        claim = make_valid_claim(claim_text="Revenue grew 12% year over year.")
        # In practice this score would come from the embedding model;
        # here we simulate the documented failure case directly.
        simulated_high_similarity_despite_negation = 0.81
        claim.mark_grounded(
            grounding_score=simulated_high_similarity_despite_negation,
            evidence="Revenue fell 12% year over year.",
            threshold_grounded=0.75,
            threshold_challenged=0.55,
        )
        assert claim.verdict == GroundingVerdict.GROUNDED  # known limitation, mitigated by Debate Agent in practice
