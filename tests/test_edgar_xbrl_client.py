"""
Unit tests for verifyfin.data_pipeline.edgar_xbrl_client.

All SEC network calls are mocked -- these tests run offline.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from verifyfin.data_pipeline.edgar_xbrl_client import EdgarXbrlClient, XbrlFactsUnavailableError

# A trimmed-down but structurally realistic sample of what
# data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json actually returns.
FAKE_COMPANY_FACTS = {
    "cik": 320193,
    "entityName": "Apple Inc.",
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        {"end": "2023-12-30", "val": 119575000000, "fy": 2024, "fp": "Q1", "form": "10-Q", "filed": "2024-02-02", "accn": "0000320193-24-000006"},
                        {"end": "2024-06-29", "val": 85777000000, "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-02", "accn": "0000320193-24-000081"},
                        {"end": "2024-03-15", "val": 90000000000, "fy": 2024, "fp": "Q2", "form": "8-K", "filed": "2024-03-20", "accn": "0000320193-24-000050"},  # 8-K should be ignored
                    ]
                },
            },
            "EarningsPerShareBasic": {
                "label": "EPS Basic",
                "units": {
                    "USD/shares": [
                        {"end": "2024-06-29", "val": 1.42, "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-02", "accn": "0000320193-24-000081"},
                    ]
                },
            },
            "Assets": {
                "label": "Total Assets",
                "units": {
                    "USD": [
                        {"end": "2024-06-29", "val": 331612000000, "fy": 2024, "fp": "Q3", "form": "10-Q", "filed": "2024-08-02", "accn": "0000320193-24-000081"},
                    ]
                },
            },
            # NetIncomeLoss deliberately absent -- tests graceful None handling
        }
    },
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unused-for-this-test")
    c = EdgarXbrlClient()
    c.cache_dir = tmp_path / "edgar_xbrl_cache"
    c.cache_dir.mkdir()
    return c


class TestGetCompanyFacts:
    def test_fetches_and_caches(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = FAKE_COMPANY_FACTS

        with patch.object(client._edgar, "get_cik_for_ticker", return_value="0000320193"), \
             patch.object(client._edgar, "_get", return_value=mock_response) as mock_get:
            data1 = client.get_company_facts("AAPL")
            data2 = client.get_company_facts("AAPL")  # second call should hit disk cache

        assert data1["entityName"] == "Apple Inc."
        assert mock_get.call_count == 1

    def test_no_facts_raises(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"cik": 1, "facts": {}}
        with patch.object(client._edgar, "get_cik_for_ticker", return_value="0000000001"), \
             patch.object(client._edgar, "_get", return_value=mock_response):
            with pytest.raises(XbrlFactsUnavailableError):
                client.get_company_facts("FOREIGNCO")

    def test_stale_cache_triggers_refetch(self, client):
        """
        Regression test for a real bug found during development: XBRL
        company facts were cached indefinitely with no expiry. When a
        new quarterly filing appeared on SEC's side between sessions,
        the stale cache kept serving the OLD quarter's figures while
        the knowledge base (built from a fresh, uncached filing lookup)
        moved on to the new quarter -- causing every numeric claim to
        fail verification, since the exact old figures no longer
        appeared anywhere in the newer indexed filing. A cache older
        than the TTL must trigger a real refetch.
        """
        cik10 = "0000320193"
        with patch.object(client._edgar, "get_cik_for_ticker", return_value=cik10):
            cache_path = client._cache_path(cik10)
            stale_payload = {
                "cached_at": (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
                "data": FAKE_COMPANY_FACTS,
            }
            with open(cache_path, "w") as f:
                json.dump(stale_payload, f)

            mock_response = MagicMock()
            mock_response.json.return_value = FAKE_COMPANY_FACTS
            with patch.object(client._edgar, "_get", return_value=mock_response) as mock_get:
                client.get_company_facts("AAPL")
                assert mock_get.call_count == 1  # stale cache correctly triggered a real refetch

    def test_fresh_cache_does_not_refetch(self, client):
        cik10 = "0000320193"
        with patch.object(client._edgar, "get_cik_for_ticker", return_value=cik10):
            cache_path = client._cache_path(cik10)
            fresh_payload = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "data": FAKE_COMPANY_FACTS,
            }
            with open(cache_path, "w") as f:
                json.dump(fresh_payload, f)

            with patch.object(client._edgar, "_get") as mock_get:
                data = client.get_company_facts("AAPL")
                assert mock_get.call_count == 0  # fresh cache correctly avoided a network call
                assert data == FAKE_COMPANY_FACTS


class TestConceptExtraction:
    def test_picks_most_recent_10q_ignoring_8k(self, client):
        entry = client._extract_concept(FAKE_COMPANY_FACTS, "revenue")
        assert entry is not None
        assert entry["form"] == "10-Q"
        assert entry["period_end"] == "2024-06-29"  # most recent 10-Q, NOT the later-dated 8-K
        assert entry["value"] == 85777000000
        assert entry["accession_number"] == "0000320193-24-000081"

    def test_missing_concept_returns_none_not_error(self, client):
        entry = client._extract_concept(FAKE_COMPANY_FACTS, "net_income")
        assert entry is None

    def test_eps_extracted_with_correct_unit(self, client):
        entry = client._extract_concept(FAKE_COMPANY_FACTS, "eps_basic")
        assert entry["value"] == 1.42
        assert entry["unit"] == "USD/shares"

    def test_regression_picks_current_tag_over_stale_discontinued_tag(self, client):
        """
        Regression test for a real bug found during development: AAPL
        reported revenue under the old 'Revenues' tag through 2018, then
        switched to 'RevenueFromContractWithCustomerExcludingAssessedTax'
        after ASC 606 took effect. The old tag still has (stale) entries.
        The extractor must return the globally most recent entry across
        ALL alias tags, not just whichever alias tag happens to be tried
        first in the alias list.
        """
        facts_with_tag_migration = {
            "facts": {
                "us-gaap": {
                    "Revenues": {  # old, discontinued tag -- tried first in the alias list
                        "units": {
                            "USD": [
                                {"end": "2018-09-29", "val": 265595000000, "form": "10-K", "filed": "2018-11-05", "accn": "old-accn"},
                            ]
                        }
                    },
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {  # current tag -- tried second
                        "units": {
                            "USD": [
                                {"start": "2025-12-29", "end": "2026-03-28", "val": 95000000000, "form": "10-Q", "filed": "2026-05-01", "accn": "current-accn"},
                            ]
                        }
                    },
                }
            }
        }
        entry = client._extract_concept(facts_with_tag_migration, "revenue")
        assert entry["period_end"] == "2026-03-28"  # must NOT be the stale 2018 entry
        assert entry["xbrl_tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
        assert entry["value"] == 95000000000

    def test_regression_picks_quarterly_over_six_month_figure_at_same_end_date(self, client):
        """
        Regression test for a real bug found during development (caught
        by VERIFY-FIN's own Debate Agent -- a nice validation of the
        project's purpose): a company's Q2 10-Q reports BOTH its 3-month
        (quarterly) and 6-month (year-to-date) revenue under the exact
        same XBRL tag, with the exact same period-end date -- only the
        period-START date differs. Selecting "most recent by end date"
        alone is ambiguous between these; the actual bug caused a claim
        that said "for the quarter" to be sourced from the six-month
        figure instead. Duration-based disambiguation must pick the
        entry closest to 91 days (one quarter), not the 6-month one.
        """
        facts_with_dual_periods = {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                # Six-month, year-to-date figure -- same end date, longer duration.
                                {"start": "2025-10-01", "end": "2026-03-28", "val": 254940000000, "form": "10-Q", "filed": "2026-05-01", "accn": "accn-1"},
                                # Three-month, quarterly figure -- same end date, ~91-day duration.
                                {"start": "2025-12-29", "end": "2026-03-28", "val": 111184000000, "form": "10-Q", "filed": "2026-05-01", "accn": "accn-1"},
                            ]
                        }
                    },
                }
            }
        }
        entry = client._extract_concept(facts_with_dual_periods, "revenue")
        assert entry["value"] == 111184000000  # the QUARTERLY figure, not the six-month figure
        assert entry["period_start"] == "2025-12-29"

    def test_instant_type_facts_unaffected_by_duration_disambiguation(self, client):
        """Balance-sheet items (total assets, liabilities, equity) have no start date and no duration ambiguity."""
        facts_instant = {
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "units": {"USD": [{"end": "2026-03-28", "val": 371082000000, "form": "10-Q", "filed": "2026-05-01", "accn": "accn-1"}]}
                    }
                }
            }
        }
        entry = client._extract_concept(facts_instant, "total_assets")
        assert entry["value"] == 371082000000


class TestGetLatestFundamentals:
    def test_returns_all_concepts_with_none_for_missing(self, client):
        with patch.object(client, "get_company_facts", return_value=FAKE_COMPANY_FACTS):
            result = client.get_latest_fundamentals("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["revenue"]["value"] == 85777000000
        assert result["net_income"] is None  # not present in fake data -- must not raise
        assert result["total_assets"]["value"] == 331612000000
