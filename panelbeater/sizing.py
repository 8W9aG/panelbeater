"""Sizing utility functions."""

# pylint: disable=too-many-locals
import numpy as np
import pandas as pd

from .simulate import SIMULATION_COLUMN


def prepare_path_matrix(sim_df, ticker_symbol):
    """Pivots Long-format simulation into Wide-format paths."""
    if not isinstance(sim_df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(sim_df)}")

    if SIMULATION_COLUMN not in sim_df.columns:
        raise KeyError(
            f"Missing 'simulation' column. Available: {list(sim_df.columns)}"
        )

    column_name = f"PX_{ticker_symbol}"
    if column_name not in sim_df.columns:
        # Flexible matching for PX_ prefix
        matches = [c for c in sim_df.columns if ticker_symbol in c and "PX_" in c]
        if not matches:
            raise KeyError(f"Could not find price column for {ticker_symbol}")
        column_name = matches[0]

    return sim_df.pivot(columns=SIMULATION_COLUMN, values=column_name)


def calculate_path_aware_mean_variance(
    path_matrix, spot_price, is_long, tp_level, sl_level
):
    """Core sizing math separated from Pandas for testability."""
    path_outcomes = []

    # Iterate through columns (paths)
    for col in range(path_matrix.shape[1]):
        single_path = path_matrix[:, col]

        if is_long:
            hit_tp = np.where(single_path >= tp_level)[0]
            hit_sl = np.where(single_path <= sl_level)[0]
        else:
            hit_tp = np.where(single_path <= tp_level)[0]
            hit_sl = np.where(single_path >= sl_level)[0]

        first_tp = hit_tp[0] if len(hit_tp) > 0 else float("inf")
        first_sl = hit_sl[0] if len(hit_sl) > 0 else float("inf")

        if first_tp < first_sl:
            path_outcomes.append((tp_level - spot_price) / spot_price)
        elif first_sl < first_tp:
            path_outcomes.append((sl_level - spot_price) / spot_price)
        else:
            path_outcomes.append((single_path[-1] - spot_price) / spot_price)

    returns = np.array(path_outcomes)
    actual_returns = returns if is_long else -returns

    mean_r = np.mean(actual_returns)
    var_r = np.var(actual_returns)

    # Add a tiny epsilon to variance to prevent division by zero/safety zeros
    # This represents 'Model Uncertainty' that never goes to zero
    epsilon_var = 1e-6

    if mean_r > 0:
        kelly = mean_r / (var_r + epsilon_var)
    else:
        kelly = 0

    return kelly, mean_r, var_r, actual_returns
