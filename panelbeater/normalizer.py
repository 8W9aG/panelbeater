"""Normalize the Y targets to standard deviations."""

# pylint: disable=too-many-locals

import math

import numpy as np
import pandas as pd
import tqdm
from wavetrainer.model.model import PROBABILITY_COLUMN_PREFIX


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataframe per column by z-score bucketing."""
    df = df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
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


def denormalize(df: pd.DataFrame) -> pd.DataFrame:
    """Denormalize the dataframe back to a total value."""
    date_to_add = df.index[-1] + pd.Timedelta(days=1)

    cols = set(df.columns.values.tolist())
    target_cols = {x for x in cols if "_".join(x.split("_")[:-1])}
    for col in target_cols:
        df[col] = None

        # Find the standard deviations
        z_cols = {x for x in cols if x.startswith(col)}
        stds = sorted([float(x.replace(col, "").split("_")[1]) for x in z_cols])

        # Find the highest probability standard deviation
        highest_std_value = 0.0
        highest_std = None
        for std in stds:
            std_suffix = f"{col}_{std}_{PROBABILITY_COLUMN_PREFIX}"
            std_true_col = sorted([x for x in cols if x.startswith(std_suffix)])[-1]
            std_value = df[std_true_col].iloc[-1]
            if std_value > highest_std_value:
                highest_std_value = std_value
                highest_std = std

        # Convert the standard deviation back to a value
        mu = df[col].rolling(365).mean()
        sigma = df[col].rolling(365).std()
        value = (highest_std * sigma) + mu
        df.loc[date_to_add, col] = df[col].iloc[-1] * (1.0 + value)

    return df.drop(columns=list(cols))
