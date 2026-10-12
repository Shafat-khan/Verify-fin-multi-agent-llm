"""
Unit tests for verifyfin.data_pipeline.fred_client.

The fredapi library is mocked -- these tests run offline and don't
require a real FRED_API_KEY.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from verifyfin.data_pipeline.fred_client import FredClient, FredSeriesUnavailableError


def _fake_series():
    dates = pd.date_range("2024-01-01", periods=3, freq="ME")
    return pd.Series([5.25, 5.33, 5.33], index=dates)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fake-key-for-offline-tests")
    with patch("verifyfin.data_pipeline.fred_client.Fred") as mock_fred_cls:
        mock_instance = MagicMock()
        mock_fred_cls.return_value = mock_instance
        c = FredClient()
        c.cache_dir = tmp_path / "fred_cache"
        c.cache_dir.mkdir()
        c._fred = mock_instance
        yield c, mock_instance


class TestGetSeries:
    def test_returns_json_safe_dict_with_latest_value(self, client):
        c, mock_instance = client
        mock_instance.get_series.return_value = _fake_series()

        data = c.get_series("FEDFUNDS")
        assert data["series_id"] == "FEDFUNDS"
        assert data["latest_value"] == 5.33
        assert len(data["dates"]) == 3

    def test_second_call_uses_cache(self, client):
        c, mock_instance = client
        mock_instance.get_series.return_value = _fake_series()

        c.get_series("FEDFUNDS")
        c.get_series("FEDFUNDS")
        assert mock_instance.get_series.call_count == 1

    def test_empty_series_raises(self, client):
        c, mock_instance = client
        mock_instance.get_series.return_value = pd.Series(dtype=float)

        with pytest.raises(FredSeriesUnavailableError):
            c.get_series("MADEUPSERIES")

    def test_api_exception_wrapped_in_specific_error(self, client):
        c, mock_instance = client
        mock_instance.get_series.side_effect = ValueError("Bad Request")

        with pytest.raises(FredSeriesUnavailableError):
            c.get_series("BADSERIES")


class TestMacroSnapshot:
    def test_skips_unavailable_series_without_failing_whole_snapshot(self, client):
        c, mock_instance = client

        def side_effect(series_id, observation_start=None):
            if series_id == "GOODSERIES":
                return _fake_series()
            raise ValueError("not found")

        mock_instance.get_series.side_effect = side_effect

        snapshot = c.get_macro_snapshot(["GOODSERIES", "BADSERIES"])
        assert "GOODSERIES" in snapshot
        assert "BADSERIES" not in snapshot


class TestValidation:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "your_fred_api_key_here")
        with pytest.raises(ValueError, match="FRED_API_KEY"):
            FredClient()
