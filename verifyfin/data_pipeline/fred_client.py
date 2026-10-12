"""
verifyfin.data_pipeline.fred_client
======================================
Thin wrapper around the `fredapi` library for macroeconomic time-series
data (Proposal Objective 2.3, Macro Agent): Federal Funds Rate, CPI,
unemployment rate, GDP, etc.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fredapi import Fred

from verifyfin.utils.io_utils import load_config, resolve_path
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FredSeriesUnavailableError(Exception):
    """Raised when FRED has no data for a requested series id."""


class FredClient:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        fred_cfg = self.config["data_sources"]["fred"]
        self.default_series = fred_cfg.get(
            "series", ["FEDFUNDS", "CPIAUCSL", "UNRATE", "GDP"]
        )

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key or api_key == "your_fred_api_key_here":
            raise ValueError(
                "FRED_API_KEY is not set or is still the placeholder value. "
                "Copy .env.example to .env and add your real key "
                "(get one free at https://fred.stlouisfed.org/docs/api/api_key.html)."
            )

        self._fred = Fred(api_key=api_key)
        self.cache_dir = resolve_path("data/raw/fred_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_hours = 24  # macro series update infrequently (monthly/quarterly); daily refresh is generous

    # ---------- caching helpers ----------

    def _cache_path(self, series_id: str) -> Path:
        return self.cache_dir / f"{series_id}.json"

    def _read_cache(self, series_id: str) -> Optional[dict]:
        path = self._cache_path(series_id)
        if not path.exists():
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        cached_at = datetime.fromisoformat(payload["cached_at"])
        if datetime.now(timezone.utc) - cached_at > timedelta(hours=self.cache_ttl_hours):
            return None
        return payload["data"]

    def _write_cache(self, series_id: str, data: dict) -> None:
        payload = {"cached_at": datetime.now(timezone.utc).isoformat(), "data": data}
        with open(self._cache_path(series_id), "w") as f:
            json.dump(payload, f, default=str)

    # ---------- public API ----------

    def get_series(self, series_id: str, lookback_days: int = 730) -> dict:
        """
        Returns a single FRED series as a JSON-safe dict:
        {"series_id": "FEDFUNDS", "dates": [...], "values": [...],
         "latest_value": ..., "latest_date": ...}
        """
        cached = self._read_cache(series_id)
        if cached is not None:
            return cached

        logger.info(f"Fetching FRED series {series_id}...")
        start_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        try:
            series = self._fred.get_series(series_id, observation_start=start_date)
        except Exception as e:
            raise FredSeriesUnavailableError(f"FRED returned an error for series '{series_id}': {e}") from e

        series = series.dropna()
        if series.empty:
            raise FredSeriesUnavailableError(f"FRED series '{series_id}' returned no data in the requested window.")

        data = {
            "series_id": series_id,
            "dates": [d.strftime("%Y-%m-%d") for d in series.index],
            "values": [round(float(v), 4) for v in series.values],
            "latest_value": round(float(series.iloc[-1]), 4),
            "latest_date": series.index[-1].strftime("%Y-%m-%d"),
        }
        self._write_cache(series_id, data)
        return data

    def get_macro_snapshot(self, series_ids: Optional[list[str]] = None) -> dict:
        """
        Returns the latest value of each configured macro series in one
        call, e.g. {"FEDFUNDS": {...}, "CPIAUCSL": {...}, ...}. This is
        what the Macro Agent uses to ground claims about the current
        macroeconomic environment.
        """
        series_ids = series_ids or self.default_series
        snapshot = {}
        for sid in series_ids:
            try:
                snapshot[sid] = self.get_series(sid)
            except FredSeriesUnavailableError as e:
                logger.warning(f"Skipping unavailable series {sid}: {e}")
        return snapshot
