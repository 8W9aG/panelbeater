"""Download historical data."""

import numpy as np
import pandas as pd
import requests_cache
import tqdm
import yfinance as yf
from pandas_datareader import data as fred


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
    dfs = []
    for code in tqdm.tqdm(codes, desc="Downloading macros"):
        s = fred.DataReader(code, "fred", start="2000-01-01", session=session)
        s.columns = [code]
        dfs.append(s)
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
    return levels.replace([np.inf, -np.inf], np.nan)
