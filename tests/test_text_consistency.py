"""Tests for the deterministic sentence-vs-evidence check (verifyfin.agents.text_consistency)."""

import pytest

from verifyfin.agents.text_consistency import (
    check_text_against_excerpt,
    extract_dates,
    extract_quantities,
    quantity_matches,
)

REV = 116017000000.0


class TestExtraction:
    def test_full_dollar_amount_with_commas(self):
        q = extract_quantities("Total revenue was $116,017,000,000 for the period.")
        assert len(q) == 1 and q[0].display == 116017000000.0 and q[0].decimals == 0 and q[0].scale == 1.0

    def test_scale_words_and_letters(self):
        assert extract_quantities("revenue of $116.0 billion")[0].scale == 1e9
        assert extract_quantities("revenue of $94.9B")[0].scale == 1e9
        assert extract_quantities("assets of 4,560 million")[0].scale == 1e6

    def test_percent_is_not_scaled(self):
        q = extract_quantities("unemployment was 4.3%")[0]
        assert q.percent and q.scale == 1.0 and q.display == 4.3

    def test_dates_years_and_quarter_tags_are_not_quantities(self):
        text = "In Q3 2026, on 2026-07-01 and June 27, 2026, EPS was $2.14 in FY2026."
        assert [q.display for q in extract_quantities(text)] == [2.14]

    def test_eps_keeps_displayed_decimals(self):
        q = extract_quantities("Basic earnings per share were $2.3 for the period.")[0]
        assert q.display == 2.3 and q.decimals == 1

    def test_dates_iso_and_long_form_normalised(self):
        assert extract_dates("On 2026-07-01 and for the period ended June 27, 2026.") == {"2026-07-01", "2026-06-27"}

    def test_no_dates(self):
        assert extract_dates("Revenue grew strongly.") == set()


class TestRoundingAwareMatch:
    def test_exact_full_number(self):
        assert quantity_matches(extract_quantities("$116,017,000,000")[0], REV)

    def test_rounded_in_billions_is_accepted(self):
        assert quantity_matches(extract_quantities("$116.0 billion")[0], REV)
        assert quantity_matches(extract_quantities("$116 billion")[0], REV)

    def test_one_percent_error_is_rejected_at_full_precision(self):
        assert not quantity_matches(extract_quantities("$117,177,170,000")[0], REV)

    def test_one_percent_error_is_rejected_in_billions(self):
        assert not quantity_matches(extract_quantities("$117.2 billion")[0], REV)

    def test_eps_rounding(self):
        assert quantity_matches(extract_quantities("$2.3")[0], 2.30)
        assert not quantity_matches(extract_quantities("$2.3")[0], 2.14)

    def test_digit_transposition_rejected(self):
        assert not quantity_matches(extract_quantities("$161,017,000,000")[0], REV)


class TestCheck:
    def test_clean_numerical_claim_passes(self):
        c = check_text_against_excerpt("Total revenue was $116,017,000,000 for the period ended June 27, 2026.", REV, "2026-06-27")
        assert (c.number, c.date) == ("ok", "ok") and c.consistent

    def test_wrong_number_in_sentence_is_a_mismatch(self):
        c = check_text_against_excerpt("Total revenue was $161,017,000,000 for the period ended June 27, 2026.", REV, "2026-06-27")
        assert c.number == "mismatch" and not c.consistent

    def test_wrong_year_in_sentence_is_a_date_mismatch(self):
        c = check_text_against_excerpt("Total revenue was $116,017,000,000 for the period ended June 27, 2025.", REV, "2026-06-27")
        assert c.date == "mismatch" and not c.consistent

    def test_macro_clean_and_shifted(self):
        assert check_text_against_excerpt("On 2026-07-01, the Federal Funds Rate was 3.63.", 3.63, "2026-07-01").consistent
        assert check_text_against_excerpt("On 2026-08-01, the Federal Funds Rate was 3.63.", 3.63, "2026-07-01").date == "mismatch"
        assert check_text_against_excerpt("On 2026-07-01, the Federal Funds Rate was 3.93.", 3.63, "2026-07-01").number == "mismatch"

    def test_sentence_with_no_number_or_date_is_not_penalised(self):
        c = check_text_against_excerpt("Revenue grew strongly.", REV, "2026-06-27")
        assert (c.number, c.date) == ("not_stated", "not_stated") and c.consistent

    def test_missing_excerpt_values_are_not_penalised(self):
        assert check_text_against_excerpt("Revenue was $5 billion on 2026-01-01.", None, None).consistent

    def test_one_matching_quantity_among_several_is_enough(self):
        c = check_text_against_excerpt("Revenue of $116,017,000,000 versus $98,000,000,000 a year earlier.", REV, None)
        assert c.number == "ok"
