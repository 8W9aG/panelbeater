"""Sizing utility functions."""

# pylint: disable=too-many-locals,invalid-name,too-many-arguments,too-many-positional-arguments
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

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


def calculate_distribution_exits(row, sim_df, horizon_pct=0.5):
    """
    Calculates TP/SL based on dynamic percentiles derived from
    the variance of the predicted option distribution.
    """
    date_val = row["expiry"]

    # 1. Lookup prices for specific expiry
    if date_val not in sim_df.index:
        target_lookup = pd.to_datetime(date_val)
        sim_prices = sim_df.loc[target_lookup].values
    else:
        sim_prices = sim_df.loc[date_val].values

    # 2. Time to Horizon calculation
    today = datetime.now()
    expiry_date = datetime.strptime(row["expiry"], "%Y-%m-%d")
    total_days = (expiry_date - today).days

    if total_days <= 0:
        return row["ask"], row["ask"]

    days_to_horizon = total_days * horizon_pct
    time_to_expiry_at_horizon = (total_days - days_to_horizon) / 365.0

    # 3. Simulate OPTION prices across all paths
    predicted_option_values = black_scholes_price(
        sim_prices,
        row["strike"],
        time_to_expiry_at_horizon,
        0.04,
        row["impliedVolatility"],
        row["type"],
    )

    # 4. DYNAMIC LOGIC: Adjust percentiles based on Option CV
    # CV = Standard Deviation / Mean
    opt_mean = np.mean(predicted_option_values)
    opt_std = np.std(predicted_option_values)

    # If mean is 0 (deep OTM), we can't calculate CV, so default to tightest
    if opt_mean <= 0.01:
        tp_pct, sl_pct = 75, 25
    else:
        cv = opt_std / opt_mean
        # As CV increases, we pull percentiles toward the median (50)
        # Low CV (high confidence): 90/10
        # High CV (low confidence): 70/30
        spread_modifier = np.clip(20 * cv, 5, 20)
        tp_pct = 90 - spread_modifier
        sl_pct = 10 + spread_modifier

    tp = np.percentile(predicted_option_values, tp_pct)
    sl = np.percentile(predicted_option_values, sl_pct)

    return tp, sl


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Vectorized Black-Scholes pricing for European options.

    Parameters:
    S (float or np.array): Current underlying price (or distribution of prices)
    K (float): Strike price
    T (float): Time to maturity in years (e.g., 0.5 for 6 months)
    r (float): Risk-free interest rate (e.g., 0.04 for 4%)
    sigma (float): Implied Volatility (e.g., 0.25 for 25%)
    option_type (str): 'call' or 'put'
    """
    # Ensure T is non-zero to avoid division by zero errors at expiration
    T = np.maximum(T, 1e-6)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price
