"""Generate features over a dataframe."""

import warnings

import numpy as np
import pandas as pd
import tqdm
from feature_engine.datetime import DatetimeFeatures


def _ticker_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    cols = df.columns.values.tolist()
    for col in tqdm.tqdm(cols, desc="Generating ticker features"):
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
    for lag in tqdm.tqdm(lags, desc="Generating lags"):
        dfs.append(df.shift(lag).add_suffix(f"_lag{lag}"))
    for window in tqdm.tqdm(windows, desc="Generating window features"):
        dfs.append(df.rolling(window).mean().add_suffix(f"_rmean{window}"))  # type: ignore
        dfs.append(df.rolling(window).std().add_suffix(f"_rstd{window}"))  # type: ignore
    return pd.concat(dfs, axis=1).replace([np.inf, -np.inf], np.nan)


def _dt_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Generating datetime features")
    dtf = DatetimeFeatures(features_to_extract="all", variables="index")
    return dtf.fit_transform(df)


def features(df: pd.DataFrame, windows: list[int], lags: list[int]) -> pd.DataFrame:
    """Generate features on a dataframe."""
    cols = df.columns.values.tolist()
    df = _ticker_features(df=df, windows=windows)
    df = _meta_ticker_feature(df, lags=lags, windows=windows)
    df = _dt_features(df=df)
    return df.drop(columns=cols)
