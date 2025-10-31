"""Normalize the Y targets to standard deviations."""

import math

import pandas as pd
import tqdm


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataframe per column by z-score bucketing."""
    mu = df.rolling(365).mean()
    sigma = df.rolling(365).std()
    df = ((((df - mu) / sigma) * 2.0).round() / 2.0).clip(-3, 3)
    dfs = []
    for col in tqdm.tqdm(df.columns, desc="Normalising targets"):
        for unique_val in df[col].unique():
            if math.isnan(unique_val):
                continue
            s = (df[col] == unique_val).rename(f"{col}_{unique_val}")
            dfs.append(s)
    return pd.concat(dfs, axis=1)
