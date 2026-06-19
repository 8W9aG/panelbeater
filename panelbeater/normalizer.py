"""Normalize the Y targets to standard deviations."""

# pylint: disable=too-many-locals,line-too-long,too-many-statements,too-many-branches
import math
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from wavetrainer.model.model import PROBABILITY_COLUMN_PREFIX


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def normalize(
    df: pd.DataFrame,
    horizon: int = 1,
    quantiles: int | None = None,
    flagged_tickers: list[str] | None = None,
    flagged_macros: list[str] | None = None,
) -> pd.DataFrame:
    """
    Normalize the dataframe per column by z-score bucketing or quantiles.

    Args:
        df: Input dataframe of prices.
        horizon: The period over which to calculate the percentage change.
        quantiles: Number of quantile bins to generate.
        flagged_tickers: List of tickers to apply quantile logic to.
        flagged_macros: List of macros to apply quantile logic to.
    """

    # 1. Calculate Percent Change for the specific Horizon
    df_pct = df.pct_change(periods=horizon, fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )

    flagged = set((flagged_tickers or []) + (flagged_macros or []))
    dfs = []

    for col in df_pct.columns:
        # Strip standard prefixes just for the flagged check
        base_col = col
        for prefix in ["PX_", "MACRO_"]:
            if base_col.startswith(prefix):
                base_col = base_col[len(prefix) :]
                break  # Only strip one prefix

        # --- NEW: QUANTILE LOGIC FOR FLAGGED ASSETS ---
        if quantiles is not None and base_col in flagged:
            # Get rolling rank (0.0 to 1.0)
            rolling_pct = df_pct[col].rolling(365).rank(pct=True)

            # Map percentile to a bucket integer (0 to quantiles-1)
            q_bins = (rolling_pct * quantiles).apply(np.floor).clip(0, quantiles - 1)

            for q in range(quantiles):
                s = (q_bins == q).rename(f"{col}_q{q}")
                dfs.append(s)

        # --- ORIGINAL: Z-SCORE LOGIC FOR UNFLAGGED ASSETS ---
        else:
            # 2. Calculate Statistics based on that specific horizon's returns
            mu = df_pct[col].rolling(365).mean()
            sigma = df_pct[col].rolling(365).std()

            # 3. Create the Normalized/Bucketed DataFrame (Z-scores)
            df_norm_col = ((((df_pct[col] - mu) / sigma) * 2.0).round() / 2.0).clip(
                -3, 3
            )

            for unique_val in df_norm_col.unique():
                if math.isnan(unique_val):
                    continue
                unique_val = unique_val + 0.0
                s = (df_norm_col == unique_val).rename(f"{col}_{unique_val}")
                dfs.append(s)

        # B. Create the specific 'null' column
        s_null = pd.Series(
            np.isclose(df_pct[col], 0.0), index=df_pct.index, name=f"{col}_null"
        )
        dfs.append(s_null)

    return pd.concat(dfs, axis=1)


def denormalize(
    df: pd.DataFrame,
    y: pd.DataFrame,
    u_sample: np.ndarray | None = None,
    quantiles: int | None = None,
    precomputed_kdes: dict | None = None,
) -> pd.DataFrame:
    """Denormalize the dataframe back to a total value."""
    df = df.reindex(y.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PerformanceWarning)
        for col in y.columns:
            df[col] = y[col]
    date_to_add = df.index[-1] + pd.Timedelta(days=1)

    cols = set(df.columns.values.tolist())
    target_cols = {"_".join(x.split("_")[:2]) for x in cols}
    asset_idx = 0
    for col in target_cols:
        historical_series = y[col].pct_change().dropna()

        # Check if this asset is using the new Quantile format
        is_quantile = any(x.startswith(f"{col}_q") for x in cols)

        if is_quantile and quantiles is not None:
            # --- QUANTILE INFERENCE RECONSTRUCTION ---
            probs = []
            q_list = list(range(quantiles))
            for q in q_list:
                prob_col = sorted(
                    [
                        x
                        for x in cols
                        if x.startswith(f"{col}_q{q}_{PROBABILITY_COLUMN_PREFIX}")
                    ]
                )[-1]
                probs.append(df[prob_col].dropna().iloc[-1])

            probs = np.array(probs) / np.sum(probs)

            # Select bucket using Inverse Transform Sampling or random choice
            if u_sample is not None and asset_idx < len(u_sample):
                cumulative_probs = np.cumsum(probs)
                idx = np.searchsorted(cumulative_probs, u_sample[asset_idx])
                sampled_q = q_list[min(idx, len(q_list) - 1)]
                asset_idx += 1
            else:
                sampled_q = np.random.choice(q_list, p=probs)

            # --- FAST PRECOMPUTED KDE SAMPLING ---
            value = None
            if (
                precomputed_kdes
                and col in precomputed_kdes
                and precomputed_kdes[col].get(sampled_q) is not None
            ):
                kde_info = precomputed_kdes[col][sampled_q]
                kde = kde_info["kde"]
                min_val = kde_info["min"]
                max_val = kde_info["max"]

                # Fast rejection sampling
                for _ in range(50):
                    sample = kde.resample(1)[0][0]
                    if min_val <= sample <= max_val:
                        value = sample
                        break

                # If rejection sampling fails, fallback to uniform
                if value is None:
                    value = np.random.uniform(
                        kde_info["fallback_min"], kde_info["fallback_max"]
                    )
            else:
                # Absolute fallback if no KDE was computed (sparse data)
                recent_history = historical_series.iloc[-365:]
                lower_bound = recent_history.quantile(sampled_q / quantiles)
                upper_bound = recent_history.quantile((sampled_q + 1) / quantiles)
                value = np.random.uniform(lower_bound, upper_bound)

        else:
            # --- ORIGINAL Z-SCORE INFERENCE RECONSTRUCTION ---
            z_cols = {
                x
                for x in cols
                if x.startswith(col)
                and x != col
                and x.split("_")[1] == col.split("_")[1]
            }
            if not z_cols:
                continue

            # Sort buckets (stds) and their associated probabilities
            stds = []
            try:
                stds = sorted(
                    [
                        float(x.replace(col, "").split("_")[1])
                        for x in z_cols
                        if _is_float(x.replace(col, "").split("_")[1])
                    ]
                )
            except IndexError as exc:
                print(f"Failed to get stds of ticker: {col}")
                raise exc

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
            mu = 0.0
            sigma = 0.0
            try:
                mu = float(historical_series.rolling(365).mean().fillna(0.0).iloc[-1])  # pyright: ignore
                sigma = float(historical_series.rolling(365).std().fillna(0.0).iloc[-1])
            except IndexError as exc:
                print(f"Failed to get mu/sigma of ticker: {col} - {historical_series}")
                raise exc

            lower_bound = highest_std - 0.25
            upper_bound = highest_std + 0.25
            jittered_std = np.random.uniform(lower_bound, upper_bound)
            value = (jittered_std * sigma) + mu

        # --- NULL CHECK (APPLIES TO BOTH) ---
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
