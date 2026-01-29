"""Generate features over a dataframe."""

import warnings

import numpy as np
import pandas as pd
from feature_engine.datetime import DatetimeFeatures


def _ticker_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    cols = df.columns.values.tolist()
    for col in cols:
        s = df[col]
        for w in windows:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                # SMA
                sma = s.rolling(w).mean()
                df[f"{col}_sma_{w}"] = sma / s - 1
                # PCT
                df[f"{col}_pctchg_{w}"] = s.pct_change(w, fill_method=None)
                # Z-Score
                mu = s.rolling(w).mean()
                sigma = s.rolling(w).std()
                df[f"{col}_z_{w}"] = (s - mu) / sigma
    return df


def _meta_ticker_feature(
    df: pd.DataFrame, lags: list[int], windows: list[int]
) -> pd.DataFrame:
    dfs = [df]
    for lag in lags:
        dfs.append(df.shift(lag).add_suffix(f"_lag{lag}"))
    for window in windows:
        dfs.append(df.rolling(window).mean().add_suffix(f"_rmean{window}"))  # type: ignore
        dfs.append(df.rolling(window).std().add_suffix(f"_rstd{window}"))  # type: ignore
    return pd.concat(dfs, axis=1).replace([np.inf, -np.inf], np.nan)


def _dt_features(df: pd.DataFrame) -> pd.DataFrame:
    dtf = DatetimeFeatures(features_to_extract="all", variables="index")
    return dtf.fit_transform(df)


def _cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates relative strength features across the panel at each timestamp.
    """
    # 1. Cross-Sectional Rank (0.0 to 1.0)
    # Good for Neural Networks/KANs as it creates a bounded, robust distribution
    # uniform across time, handling outliers well.
    # FIX: Rank based on 1-day returns (Relative Strength), not nominal price
    # "Who is winning today?"
    daily_returns = df.pct_change(fill_method=None)
    cs_rank = daily_returns.rank(axis=1, pct=True).add_suffix("_xs_rank")

    # 2. Cross-Sectional Z-Score (Distance from Market Mean)
    # Measures how many standard deviations an asset is from the daily panel mean.
    # Effectively removes the 'Market Factor' to focus on relative performance.
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    # Broadcast subtraction/division across columns
    cs_z = df.sub(mean, axis=0).div(std, axis=0)
    cs_z = cs_z.add_suffix("_xs_z")

    return pd.concat([cs_rank, cs_z], axis=1)


def features(df: pd.DataFrame, windows: list[int], lags: list[int]) -> pd.DataFrame:
    """Generate features on a dataframe."""
    # 1. Deduplicate inputs to prevent column collisions
    windows = sorted(list(set(windows)))
    lags = sorted(list(set(lags)))

    original_cols = df.columns.values.tolist()

    # 2. Generate Cross-Sectional Features (Use ONLY original cols)
    # We do this first on the raw data to ensure we are ranking
    # the assets against each other, not against their own SMAs.
    df_xs = _cross_sectional_features(df[original_cols])  # pyright: ignore

    # 3. Generate Time-Series Features (modifies df in-place)
    df = _ticker_features(df=df, windows=windows)

    # 4. Generate Meta Features (Lags/Rolling of the expanded df)
    # Note: This will generate lags for the SMAs calculated in step 3 as well.
    df = _meta_ticker_feature(df, lags=lags, windows=windows)

    # 5. Merge Cross-Sectional Features
    df = pd.concat([df, df_xs], axis=1)

    # 6. Datetime Features
    df = _dt_features(df=df)

    # 7. Final Clean up
    # Drop original raw prices (optional, but standard if differencing)
    # shift(1) ensures we don't leak the future (predicting t using t-1 data)
    return df.drop(columns=original_cols).shift(1)
