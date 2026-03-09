"""Download historical data."""

# pylint: disable=invalid-name,global-statement,unused-argument,broad-exception-caught
import os

import numpy as np
import pandas as pd
import requests_cache
import tqdm
import yfinance as yf
from fredapi import Fred  # type: ignore

_FRED_CLIENT = None


def _get_fred_client() -> Fred:
    global _FRED_CLIENT
    if _FRED_CLIENT is None:
        _FRED_CLIENT = Fred(api_key=os.environ["FRED_API_KEY"])
    return _FRED_CLIENT


def _load_yahoo_prices(tickers: list[str]) -> pd.DataFrame:
    """Adj Close for all tickers, daily."""
    print(f"Download tickers: {tickers}")
    px = yf.download(
        tickers,
        start="2000-01-01",
        end=None,
        auto_adjust=True,
        progress=False,
        proxy=None,
        session=None,
    )
    if px is None or px.empty:
        raise ValueError("px is null or empty")

    px = px["Close"]

    # Handle the case where yf returns a Series for a single ticker
    if isinstance(px, pd.Series):
        px = px.to_frame()

    if isinstance(px.columns, pd.MultiIndex):
        px = px.droplevel(0, axis=1)

    pxf = px.sort_index().astype(float)

    # FIX: Ensure index is a DatetimeIndex (not mixed with date objects)
    pxf.index = pd.to_datetime(pxf.index)
    return pxf


def _load_macro_series(
    codes: list[str], session: requests_cache.CachedSession
) -> pd.DataFrame:
    """Load macro series from FRED or Yahoo Finance, forward-fill to daily."""
    dfs: list[pd.Series] = []

    yf_codes = [c for c in codes if c.startswith("^")]
    fred_codes = [c for c in codes if not c.startswith("^")]

    # 1. Process Yahoo Finance Macros
    if yf_codes:
        print(f"Downloading Yahoo macros: {yf_codes}")
        yf_data = _load_yahoo_prices(yf_codes)
        for code in yf_codes:
            dfs.append(yf_data[code].rename(code))  # pyright: ignore

    # 2. Process FRED Macros
    if fred_codes:
        client = _get_fred_client()
        for code in tqdm.tqdm(fred_codes, desc="Downloading FRED macros"):
            try:
                df = client.get_series_all_releases(code)
                df["date"] = pd.to_datetime(df["date"])
                df["realtime_start"] = pd.to_datetime(df["realtime_start"])

                # FIX 1: Sort chronologically by observation date
                df = df.sort_values(by="date")

                # FIX 2: Find the FIRST time each observation was released (prevents lookahead bias)
                first_releases = df.loc[df.groupby("date")["realtime_start"].idxmin()]

                # FIX 3: Set the index to the REPORTING DATE, not the observation date
                first_releases = first_releases.set_index("realtime_start")

                # FIX 4: If multiple historical dates were published on the same day,
                # keep only the most recent observation for that reporting date
                first_releases = first_releases.groupby(level=0).tail(1)

                first_releases.index = pd.to_datetime(first_releases.index)
                dfs.append(first_releases["value"].rename(code))  # pyright: ignore
            except Exception:
                # Fallback to standard series (Warning: this will have lookahead bias)
                df = client.get_series(code)
                df.index = pd.to_datetime(df.index)
                dfs.append(df.rename(code))

    if not dfs:
        return pd.DataFrame()

    # Combine
    macro = pd.concat(dfs, axis=1)

    # Standardize index to DatetimeIndex before sorting
    macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()

    # asfreq("D") works cleanly with the reporting date index
    macro = macro.asfreq("D").ffill()
    return macro


def download(
    tickers: list[str], macros: list[str], session: requests_cache.CachedSession
) -> pd.DataFrame:
    """Download the historical data."""
    prices = _load_yahoo_prices(tickers=tickers)
    macro = _load_macro_series(codes=macros, session=session)
    idx = prices.index.union(macro.index)
    prices = prices.reindex(idx).ffill()
    macro = macro.reindex(idx).ffill()
    prices_min = prices.dropna(how="all").index.min()
    macro_min = macro.dropna(how="all").index.min()
    common_start = max(prices_min, macro_min)  # type: ignore
    prices = prices.loc[common_start:]
    macro = macro.loc[common_start:]
    levels = pd.concat(
        [prices.add_prefix("PX_"), macro.add_prefix("MACRO_")], axis=1
    ).ffill()
    print(levels)
    return levels.replace([np.inf, -np.inf], np.nan)
