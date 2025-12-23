"""Download historical data."""

# pylint: disable=invalid-name,global-statement,unused-argument
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
    )
    if px is None:
        raise ValueError("px is null")
    if not isinstance(px, pd.DataFrame):
        raise ValueError("px is not a dataframe")
    px = px["Close"]
    if isinstance(px.columns, pd.MultiIndex):
        px = px.droplevel(0, axis=1)
    pxf = px.sort_index().astype(float)
    if not isinstance(pxf, pd.DataFrame):
        raise ValueError("pxf is not a dataframe")
    return pxf


def _load_fred_series(
    codes: list[str], session: requests_cache.CachedSession
) -> pd.DataFrame:
    """Load FRED series, forward-fill to daily to align with markets."""
    client = _get_fred_client()
    dfs: list[pd.Series] = []
    for code in tqdm.tqdm(codes, desc="Downloading macros"):
        try:
            df = client.get_series_all_releases(code)
            df["date"] = pd.to_datetime(df["date"])
            df["realtime_start"] = pd.to_datetime(df["realtime_start"])

            def select_latest(group: pd.DataFrame) -> pd.DataFrame:
                latest_df = group[
                    group["realtime_start"] == group["realtime_start"].max()
                ]
                if not isinstance(latest_df, pd.DataFrame):
                    raise ValueError("latest_df is not a DataFrame")
                return latest_df

            df = df.groupby("date").apply(select_latest)
            df = df.set_index("date")
            df.index = df.index.date  # type: ignore
            df = df.sort_index()
            dfs.append(df["value"].rename(code))  # type: ignore
        except ValueError:
            df = client.get_series(code)
            df.index = df.index.date  # type: ignore
            df = df.sort_index()
            dfs.append(df.rename(code))
    macro = pd.concat(dfs, axis=1).sort_index()
    # daily frequency with forward-fill (macro is slower cadence)
    macro = macro.asfreq("D").ffill()
    return macro


def download(
    tickers: list[str], macros: list[str], session: requests_cache.CachedSession
) -> pd.DataFrame:
    """Download the historical data."""
    prices = _load_yahoo_prices(tickers=tickers)
    macro = _load_fred_series(codes=macros, session=session)
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
