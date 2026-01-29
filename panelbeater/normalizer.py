"""Normalize the Y targets to standard deviations."""

# pylint: disable=too-many-locals
import math

import numpy as np
import pandas as pd
from wavetrainer.model.model import PROBABILITY_COLUMN_PREFIX


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataframe per column by z-score bucketing."""

    # 1. Calculate Percent Change (preserve this in a specific variable)
    df_pct = df.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    # 2. Calculate Statistics based on pct_change
    mu = df_pct.rolling(365).mean()
    sigma = df_pct.rolling(365).std()

    # 3. Create the Normalized/Bucketed DataFrame (Z-scores)
    # We use df_pct here, not the original df
    df_norm = ((((df_pct - mu) / sigma) * 2.0).round() / 2.0).clip(-3, 3)

    dfs = []
    for col in df_norm.columns:
        # A. Create the bucket columns (based on Z-score)
        for unique_val in df_norm[col].unique():
            if math.isnan(unique_val):
                continue
            s = (df_norm[col] == unique_val).rename(f"{col}_{unique_val}")
            dfs.append(s)

        # B. Create the specific 'null' column (based on raw pct_change)
        # We look at df_pct to see if the change was exactly 0.0
        s_null = pd.Series(
            np.isclose(df_pct[col], 0.0), index=df_pct.index, name=f"{col}_null"
        )
        dfs.append(s_null)

    return pd.concat(dfs, axis=1)


def denormalize(
    df: pd.DataFrame, y: pd.DataFrame, u_sample: np.ndarray | None = None
) -> pd.DataFrame:
    """Denormalize the dataframe back to a total value."""
    df = df.reindex(y.index)
    for col in y.columns:
        df[col] = y[col]
    date_to_add = df.index[-1] + pd.Timedelta(days=1)

    cols = set(df.columns.values.tolist())
    target_cols = {"_".join(x.split("_")[:2]) for x in cols}
    asset_idx = 0
    for col in target_cols:
        # 1. Gather all predicted probabilities for this asset's buckets
        z_cols = {x for x in cols if x.startswith(col) and x != col}
        if not z_cols:
            continue
        historical_series = y[col].pct_change().dropna()

        # Sort buckets (stds) and their associated probabilities
        stds = sorted(
            [
                float(x.replace(col, "").split("_")[1])
                for x in z_cols
                if _is_float(x.replace(col, "").split("_")[1])
            ]
        )
        probs = []
        for std in stds:
            std_suffix = f"{col}_{std}_{PROBABILITY_COLUMN_PREFIX}"
            prob_col = sorted([x for x in cols if x.startswith(std_suffix)])[-1]
            prob = df[prob_col].dropna().iloc[-1]
            probs.append(prob)

        # Normalize probabilities (ensure they sum to 1.0)
        probs = np.array(probs) / np.sum(probs)

        # 2. Select the bucket using Inverse Transform Sampling
        highest_std = 0.0
        if u_sample is not None and asset_idx < len(u_sample):
            cumulative_probs = np.cumsum(probs)
            idx = np.searchsorted(cumulative_probs, u_sample[asset_idx])
            highest_std = stds[min(idx, len(stds) - 1)]
            asset_idx += 1
        else:
            highest_std = np.random.choice(stds, p=probs)

        # 3. Use Pandas rolling on the historical y dataframe to avoid ndarray errors
        mu = float(historical_series.rolling(365).mean().fillna(0.0).iloc[-1])  # pyright: ignore
        sigma = float(historical_series.rolling(365).std().fillna(0.0).iloc[-1])

        lower_bound = highest_std - 0.25
        upper_bound = highest_std + 0.25
        jittered_std = np.random.uniform(lower_bound, upper_bound)
        value = (jittered_std * sigma) + mu

        if highest_std == 0.0:
            # Construct the null probability column name
            # Pattern: {col}_null_{PROBABILITY_COLUMN_PREFIX}
            null_prefix = f"{col}_null_{PROBABILITY_COLUMN_PREFIX}"

            # Find the matching column in 'cols'
            possible_null_cols = [x for x in cols if x.startswith(null_prefix)]

            if possible_null_cols:
                # Select the correct column (handling versions/suffixes)
                null_prob_col = sorted(possible_null_cols)[-1]

                # Get the probability of being exactly null
                null_prob = df[null_prob_col].dropna().iloc[-1]

                # Run random check
                # Note: We use standard random here to avoid consuming extra indices from u_sample
                if np.random.uniform(0, 1) < null_prob:
                    value = 0.0  # Force exact same price (0% change)

        df.loc[date_to_add, col] = y[col].iloc[-1] * (1.0 + value)

    return df[sorted(target_cols)]  # pyright: ignore
