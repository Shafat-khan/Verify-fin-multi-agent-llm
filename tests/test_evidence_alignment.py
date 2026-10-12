"""
Tests for verifyfin.agents.evidence_alignment. The filing chunks below are REAL chunks copied from the AAPL-2026-Q2
knowledge base (see outputs/results/chunk_samples.txt), so the parsing is tested against the real format.
"""

import pytest

from verifyfin.agents import evidence_alignment as ea

CH_SALES = ("Three Months Ended | Nine Months Ended | June 27,2026 | June 28,2025 | Change | June 27,2026 | June 28,2025 | Change || "
            "Total net sales | $ | 109,417 | $ | 94,036 | 16% | $ | 364,357 | $ | 313,695 | 16%")
CH_NI = ("Three Months Ended | Nine Months Ended | June 27,2026 | June 28,2025 | June 27,2026 | June 28,2025 | Numerator: || "
         "Net income | $ | 29,789 | $ | 23,434 | $ | 101,464 | $ | 84,544")
CH_DEPS = ("Three Months Ended | Nine Months Ended | June 27,2026 | June 28,2025 | June 27,2026 | June 28,2025 | Numerator: || "
           "Diluted earnings per share | $ | 2.02 | $ | 1.57 | $ | 6.88 | $ | 5.62")
CH_BEPS = ("Three Months Ended | Nine Months Ended | June 27,2026 | June 28,2025 | June 27,2026 | June 28,2025 | Numerator: || "
           "Basic earnings per share | $ | 2.03 | $ | 1.57 | $ | 6.91 | $ | 5.64")
CH_TA = "June 27,2026 | September 27,2025 | ASSETS: | Current assets: || Total assets | $ | 383,266 | $ | 359,241"
CH_TL = "June 27,2026 | September 27,2025 | ASSETS: | Current assets: || Total liabilities | 275,746 | 285,508"
CH_COVER = "Common Stock, $0.00001 par value per share | AAPL | The Nasdaq Stock Market LLC"
CH_HEADING = "Net sales:"


class TestContainsNumber:
    @pytest.mark.parametrize("text,variants,expected", [
        ("rate 4.1 on 2026-07-01", {"4.1"}, True),
        ("rate 14.1 on 2026-07-01", {"4.1"}, False),       # not inside a longer number
        ("rate 4.12 on 2026-07-01", {"4.1"}, False),
        ("sales | 109,417 | 94,036", {"109,417"}, True),
        ("sales 109,417,000 total", {"109,417"}, False),
        ("sales $109,417", {"109,417"}, True),
        ("ended 4.1.", {"4.1"}, True),                     # sentence-final full stop is not a digit
    ])
    def test_whole_token_matching(self, text, variants, expected):
        assert ea.contains_number(text, variants) is expected


class TestFredRecords:
    def test_parse(self):
        assert ea.parse_fred_chunk("FEDFUNDS observation: 3.64 on 2026-04-01") == ("FEDFUNDS", 3.64, "2026-04-01")
        assert ea.parse_fred_chunk("UNRATE observation: 4.1 on 2026-07-01") == ("UNRATE", 4.1, "2026-07-01")
        assert ea.parse_fred_chunk(CH_SALES) is None
        assert ea.parse_fred_chunk("Headline: Apple shares fall") is None

    def test_series_named_in_sentence(self):
        assert ea.claim_series("On 2026-07-01, the U.S. unemployment rate was 4.1.") == {"UNRATE"}
        assert ea.claim_series("The Federal Funds Rate was 3.64 on 2026-04-01.") == {"FEDFUNDS"}
        assert ea.claim_series("U.S. Gross Domestic Product (GDP) was 32888.66 on 2026-04-01.") == {"GDP"}
        assert ea.claim_series("The Consumer Price Index (CPI) was 332.407 on 2025-04-01.") == {"CPIAUCSL"}
        assert ea.claim_series("Revenue grew.") == set()

    def test_record_needs_exact_date_and_value(self):
        c = [{"text": "FEDFUNDS observation: 3.64 on 2026-03-01"}, {"text": "FEDFUNDS observation: 3.64 on 2026-04-01"},
             {"text": "FEDFUNDS observation: 3.63 on 2026-04-01"}]
        got = ea.fred_records(c, {"3.64"}, "2026-04-01")
        assert [g["text"] for g in got] == ["FEDFUNDS observation: 3.64 on 2026-04-01"]
        assert ea.fred_records(c, {"3.64"}, "2025-04-01") == []

    def test_integer_valued_observation(self):
        assert ea.fred_records([{"text": "UNRATE observation: 4.0 on 2026-01-01"}], {"4"}, "2026-01-01")

    def test_pick_prefers_the_series_the_sentence_names(self):
        recs = [{"text": "UNRATE observation: 3.64 on 2026-04-01"}, {"text": "FEDFUNDS observation: 3.64 on 2026-04-01"}]
        assert ea.pick_record(recs, {"FEDFUNDS"})["text"].startswith("FEDFUNDS")
        assert ea.pick_record(recs, set())["text"].startswith("UNRATE")
        assert ea.pick_record([], {"FEDFUNDS"}) is None

    def test_series_status(self):
        chunk = "FEDFUNDS observation: 3.64 on 2026-04-01"
        assert ea.series_status("The Federal Funds Rate was 3.64.", chunk) == "ok"
        assert ea.series_status("The U.S. unemployment rate was 3.64.", chunk) == "mismatch"
        assert ea.series_status("The rate was 3.64.", chunk) == "unknown"
        assert ea.series_status("The Federal Funds Rate was 3.64.", CH_SALES) == "unknown"


class TestLineItems:
    def test_sentence_concepts(self):
        assert ea.claim_concepts("Apple reported total revenue of $109,417,000,000.") == {"revenue"}
        assert ea.claim_concepts("Apple's net income was $29.8 billion.") == {"net_income"}
        assert ea.claim_concepts("Diluted earnings per share were $2.02.") == {"eps_diluted"}
        assert ea.claim_concepts("Basic earnings per share were $2.03.") == {"eps_basic"}
        assert ea.claim_concepts("Earnings per share were $2.02.") == {"eps_basic", "eps_diluted"}
        assert ea.claim_concepts("Apple's stockholders' equity was $100B.") == {"stockholders_equity"}
        assert ea.claim_concepts("Apple's shareholders’ equity was $100B.") == {"stockholders_equity"}
        assert ea.claim_concepts("Total assets were $383.3 billion.") == {"total_assets"}
        assert ea.claim_concepts("Things happened.") == set()

    def test_chunk_concepts(self):
        assert ea.chunk_concepts(CH_SALES) == {"revenue"}
        assert ea.chunk_concepts(CH_NI) == {"net_income"}
        assert ea.chunk_concepts(CH_DEPS) == {"eps_diluted"}
        assert ea.chunk_concepts(CH_TA) == {"total_assets"}
        assert ea.chunk_concepts(CH_COVER) == set()

    @pytest.mark.parametrize("claim,chunk,expected", [
        ("Apple reported total revenue of $109.4 billion.", CH_SALES, "ok"),
        ("Apple reported net income of $109.4 billion.", CH_SALES, "mismatch"),        # label swapped, number kept
        ("Apple's total liabilities were $383.3 billion.", CH_TA, "mismatch"),
        ("Apple's total assets were $383.3 billion.", CH_TA, "ok"),
        ("Basic earnings per share were $2.02.", CH_DEPS, "mismatch"),
        ("Diluted earnings per share were $2.02.", CH_DEPS, "ok"),
        ("Apple reported net income of $29.8 billion.", CH_COVER, "unknown"),          # unlabelled chunk: never a mismatch
        ("Apple reported net income of $29.8 billion.", CH_HEADING, "mismatch"),       # a heading that is clearly another item
        ("Something unrelated happened.", CH_SALES, "unknown"),
    ])
    def test_label_status(self, claim, chunk, expected):
        assert ea.label_status(claim, chunk) == expected

    def test_a_chunk_showing_both_items_is_not_a_mismatch(self):
        both = CH_SALES.split("||")[0] + "|| Total net sales | 1 | 2 || Net income | 3 | 4"
        assert ea.label_status("Apple's net income was 3.", both) == "ok"


class TestFilingColumns:
    def test_header_date_formats(self):
        assert ea.header_date("June 27,2026") == "2026-06-27"
        assert ea.header_date("Sep. 27, 2025") == "2025-09-27"
        assert ea.header_date("March 28, 2026") == "2026-03-28"
        assert ea.header_date("Change") is None

    def test_current_period_value_in_current_period_column(self):
        assert ea.period_status(CH_SALES, {"109,417", "109417"}, "2026-06-27") == "ok"
        assert ea.period_status(CH_NI, {"29,789"}, "2026-06-27") == "ok"
        assert ea.period_status(CH_DEPS, {"2.02"}, "2026-06-27") == "ok"
        assert ea.period_status(CH_TA, {"383,266"}, "2026-06-27") == "ok"

    def test_prior_period_value_carries_the_prior_date(self):
        assert ea.period_status(CH_SALES, {"94,036"}, "2025-06-28") == "ok"
        assert ea.period_status(CH_TA, {"359,241"}, "2025-09-27") == "ok"

    def test_right_number_wrong_period_is_a_mismatch(self):
        assert ea.period_status(CH_SALES, {"109,417"}, "2025-06-28") == "mismatch"       # current value, prior-year date
        assert ea.period_status(CH_SALES, {"94,036"}, "2026-06-27") == "mismatch"        # prior value, current date
        assert ea.period_status(CH_TA, {"359,241"}, "2026-06-27") == "mismatch"

    def test_nine_month_column_has_the_same_date_as_the_quarter(self):
        assert ea.period_status(CH_SALES, {"364,357"}, "2026-06-27") == "ok"   # documented limit: duration is not checked

    @pytest.mark.parametrize("chunk", [
        CH_COVER, CH_HEADING, "no separator here 109,417",
        "June 27,2026 | June 28,2025 || Total net sales | $ | 109,417",                  # 2 date columns but 1 value
        "Item | Amount || Total net sales | $ | 109,417 | $ | 94,036",                  # no dates in the header
    ])
    def test_unalignable_chunks_are_unknown_never_a_mismatch(self, chunk):
        assert ea.period_status(chunk, {"109,417"}, "2025-06-28") == "unknown"

    def test_number_absent_from_every_row_is_unknown(self):
        assert ea.period_status(CH_SALES, {"999"}, "2026-06-27") == "unknown"

    def test_no_claim_date_is_unknown(self):
        assert ea.period_status(CH_SALES, {"109,417"}, None) == "unknown"


# ---- real chunks that the first v3 wrongly flagged (correct claims, found by check_v3_on_real_claims.py) ----
CH_MSFT_BEPS = ("(In millions, except per share amounts) (Unaudited) | Three Months EndedMarch 31, | Nine Months EndedMarch 31, | "
                "2026 | 2025 | 2026 | 2025 | Revenue: || Basic | $ | 4.28 | $ | 3.47 | $ | 13.19 | $ | 10.03")
CH_JPM_EPS = ("(in millions, except per share amounts) | Three months ended March 31, | 2026 | 2025 | Basic earnings per share || "
              "Net income per share | $ | 5.94 | $ | 5.07")
CH_XOM_DEPS = ("(millions of dollars, unless noted) | Note Reference Number | Three Months EndedJune 30, | Six Months EndedJune 30, | "
               "2026 | 2025 | 2026 | 2025 | Revenues and other income || Earnings (loss) per common share - assuming dilution (dollars) | 2 | 3.48 | 1.64 | 4.47 | 3.40")


class TestRealFilingLabels:
    def test_msft_bare_basic_row_is_eps_not_revenue(self):
        # the header's 'Revenue:' is a section heading, not the line item of the 'Basic' row
        assert ea.chunk_concepts(CH_MSFT_BEPS) == {"eps_basic"}
        assert ea.label_status("Microsoft's basic earnings per share were $4.28 for the quarter ended March 31, 2026.", CH_MSFT_BEPS) == "ok"

    def test_msft_basic_row_still_mismatches_a_diluted_claim(self):
        assert ea.label_status("Microsoft reported diluted earnings per share of $4.27.", CH_MSFT_BEPS) == "mismatch"

    def test_xom_assuming_dilution_is_diluted_eps(self):
        assert ea.chunk_concepts(CH_XOM_DEPS) == {"eps_diluted"}
        assert ea.label_status("Diluted earnings per share were $3.48 for the period ended June 30, 2026.", CH_XOM_DEPS) == "ok"

    def test_jpm_net_income_per_share_is_compatible_with_basic_and_diluted(self):
        # the row does not say basic or diluted, so the check must not flag it (limit: it cannot tell the two apart here)
        assert ea.label_status("Diluted earnings per share for the quarter ended March 31, 2026 was $5.94.", CH_JPM_EPS) == "ok"

    def test_net_income_claim_is_still_flagged_against_a_per_share_row(self):
        assert ea.label_status("JPMorgan reported net income of $5.94 billion.", CH_JPM_EPS) == "mismatch"

    def test_section_heading_in_header_does_not_count(self):
        assert ea.chunk_concepts("Revenues and other income | 2026 || Total assets | 5") == {"total_assets"}
