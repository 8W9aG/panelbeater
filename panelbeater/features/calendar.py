"""Earnings and macro-release calendar distance features."""

# pylint: disable=broad-exception-caught

import os
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import yfinance as yf

_FRED_API_ROOT = "https://api.stlouisfed.org/fred"


@lru_cache(maxsize=None)
def fetch_yfinance_earnings(ticker: str, limit: int = 100) -> pd.DatetimeIndex:
    """
    Safely fetch earnings dates. Returns an empty DatetimeIndex if
    the asset is an ETF (e.g., SPY) or has no earnings history.

    Cached per-process: this depends only on the ticker, not on the price
    data or the date range being featurized, but features() is called once
    per simulated day inside run_single_simulation's loop (see simulate.py)
    -- without caching, that refetches the same calendar over the network
    on every single day-step of every simulation.
    """
    try:
        t = yf.Ticker(ticker)

        # Suppress yfinance stdout/warnings for cleaner bulk processing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates_df = t.get_earnings_dates(limit=limit)

        # Handle SPY, indices, or missing data
        if dates_df is None or dates_df.empty:
            return pd.DatetimeIndex([])

        # Strip timezone and normalize to midnight for exact date alignment
        return (
            pd.to_datetime(dates_df.index).tz_localize(None).normalize().sort_values()
        )

    except Exception:  # noqa
        # Catch internal yfinance parsing errors on unsupported assets
        return pd.DatetimeIndex([])


@lru_cache(maxsize=None)
def fetch_fred_release_dates(series_code: str, limit: int = 100) -> pd.DatetimeIndex:
    """
    Safely fetch a FRED series' release calendar (past and publicly
    scheduled future release/announcement dates, e.g. CPI or FOMC dates).
    Returns an empty DatetimeIndex if the series has no associated FRED
    release (e.g. Yahoo-sourced index macros like ^IRX) or the lookup fails.

    Cached per-process: see fetch_yfinance_earnings for why this matters.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key or series_code.startswith("^"):
        return pd.DatetimeIndex([])

    try:
        release_resp = requests.get(
            f"{_FRED_API_ROOT}/series/release",
            params={
                "series_id": series_code,
                "api_key": api_key,
                "file_type": "json",
            },
            timeout=10,
        )
        release_resp.raise_for_status()
        releases = release_resp.json().get("releases", [])
        if not releases:
            return pd.DatetimeIndex([])
        release_id = releases[0]["id"]

        dates_resp = requests.get(
            f"{_FRED_API_ROOT}/release/dates",
            params={
                "release_id": release_id,
                "api_key": api_key,
                "file_type": "json",
                "include_release_dates_with_no_data": "true",
                "limit": limit,
                "sort_order": "desc",
            },
            timeout=10,
        )
        dates_resp.raise_for_status()
        release_dates = dates_resp.json().get("release_dates", [])
        if not release_dates:
            return pd.DatetimeIndex([])

        return (
            pd.to_datetime([d["date"] for d in release_dates]).normalize().sort_values()
        )

    except Exception:  # noqa
        # Catch network/parsing errors on unsupported or misconfigured series
        return pd.DatetimeIndex([])


def _compute_calendar_distances(
    target_idx: pd.DatetimeIndex, cal_idx: pd.DatetimeIndex, prefix: str, label: str
) -> pd.DataFrame:
    """
    Vectorized computation of days since last and days until next calendar event.
    """
    # --- Edge Case: Asset has no calendar events (e.g., SPY earnings, ^IRX releases) ---
    if cal_idx.empty:
        return pd.DataFrame(
            {
                f"{prefix}_days_since_{label}": np.nan,
                f"{prefix}_days_until_{label}": np.nan,
            },
            index=target_idx,
        )

    # Prepare dataframes for merge_asof (must be sorted)
    df_target = pd.DataFrame(index=target_idx.normalize()).sort_index()
    df_cal = pd.DataFrame({"cal_date": cal_idx}, index=cal_idx).sort_index()

    # Backward look: Most recent past date
    past = pd.merge_asof(
        df_target, df_cal, left_index=True, right_index=True, direction="backward"
    )
    days_since = (df_target.index - past["cal_date"]).dt.days

    # Forward look: Next upcoming date
    future = pd.merge_asof(
        df_target, df_cal, left_index=True, right_index=True, direction="forward"
    )
    days_until = (future["cal_date"] - df_target.index).dt.days

    return pd.DataFrame(
        {
            f"{prefix}_days_since_{label}": days_since,
            f"{prefix}_days_until_{label}": days_until,
        },
        index=target_idx,
    )


def _apply_calendar_features(
    df: pd.DataFrame, original_cols: list[str], cal_dict: dict, label: str
) -> pd.DataFrame:
    cal_features = []

    for col in original_cols:
        cal_idx = cal_dict.get(col, pd.DatetimeIndex([]))

        # Calculate distances (will return NaNs for empty calendars)
        cal_df = _compute_calendar_distances(
            pd.DatetimeIndex(df.index), cal_idx, prefix=col, label=label
        )
        cal_features.append(cal_df)

    if cal_features:
        return pd.concat([df] + cal_features, axis=1)

    return df
