"""Process the options for the assets."""

# pylint: disable=too-many-locals,consider-using-f-string,use-dict-literal,invalid-name,too-many-arguments,too-many-positional-arguments,too-many-statements,line-too-long,bare-except
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
from scipy.stats import norm


def calculate_full_kelly_path_aware(row, sim_df):
    """
    Comprehensive Kelly sizing for options:
    Path-aware, variance-penalized, and uses dynamic boundaries.
    """
    entry_price = row["ask"]
    if entry_price <= 0:
        return 0, 0

    # 1. Get dynamic exits for this specific option
    tp_target, sl_target = calculate_distribution_exits(row, sim_df)

    # 2. Calculate option value path matrix
    # (Converting the entire underlying sim_df into an option value sim_df)
    # This is computationally heavier but provides the most accurate variance.
    path_matrix = sim_df.values

    # We use a vectorized Black-Scholes across the whole matrix
    # Note: We assume IV and r are constant for simplicity along the path
    # In a real model, T would decrease as we move down rows (days)
    # For now, we use terminal payoff to represent the "hit" potential

    path_outcomes = []
    for col in range(path_matrix.shape[1]):
        single_path = path_matrix[:, col]

        # Check if underlying ever makes the option worth TP or SL
        if row["type"] == "call":
            best_val = np.max(single_path) - row["strike"]
            worst_val = np.min(single_path) - row["strike"]
        else:
            best_val = row["strike"] - np.min(single_path)
            worst_val = row["strike"] - np.max(single_path)

        # Outcome mapping
        if best_val >= tp_target:
            path_outcomes.append((tp_target - entry_price) / entry_price)
        elif worst_val <= sl_target:
            path_outcomes.append((sl_target - entry_price) / entry_price)
        else:
            # Terminal outcome
            terminal_payoff = max(
                0,
                (path_matrix[-1, col] - row["strike"])
                if row["type"] == "call"
                else (row["strike"] - path_matrix[-1, col]),
            )
            path_outcomes.append((terminal_payoff - entry_price) / entry_price)

    # 3. Variance-Aware Kelly Formula
    path_returns = np.array(path_outcomes)
    mean_r = np.mean(path_returns)
    var_r = np.var(path_returns)

    if var_r > 0 and mean_r > 0:
        f_star = mean_r / var_r
    else:
        f_star = 0

    return f_star, mean_r * entry_price


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


def find_mispriced_options_comprehensive(
    ticker_symbol: str, sim_df: pd.DataFrame
) -> pd.DataFrame | None:
    """Comprehensively find mispriced options in ITM and OTM."""
    ticker = yf.Ticker(ticker_symbol)
    spot = ticker.history(period="1d")["Close"].iloc[-1]

    sim_dates = pd.to_datetime(sim_df.index).date.tolist()  # pyright: ignore
    available_expiries = [
        datetime.strptime(d, "%Y-%m-%d").date() for d in ticker.options
    ]
    common_dates = sorted(list(set(sim_dates).intersection(set(available_expiries))))

    all_results = []

    for target_date in common_dates:
        date_str = target_date.strftime("%Y-%m-%d")
        chain = ticker.option_chain(date_str)

        # Pulling the full chain for both calls and puts
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["type"] = "call"
        puts["type"] = "put"
        # 1. INJECT THE EXPIRY HERE
        # This ensures the row has the key your function is looking for
        calls["expiry"] = date_str
        puts["expiry"] = date_str

        full_chain = pd.concat([calls, puts])

        # --- LIQUIDITY FILTER START ---
        # Filter for options that actually trade.
        # Threshold: Open Interest > 50 AND Volume > 5 (Adjust as needed)
        full_chain = full_chain[
            (full_chain["openInterest"] > 50)
            & (full_chain["volume"] >= 5)  # Change to > 0 for strictly active today
            & (full_chain["ask"] > 0.05)  # Ignore "worthless" deep OTM scrap
        ].copy()
        # --- LIQUIDITY FILTER END ---

        model_prices_at_t = sim_df.loc[date_str].values

        for _, row in full_chain.iterrows():
            k = row["strike"]
            ask = row["ask"]
            bid = row["bid"]
            symbol = row[
                "contractSymbol"
            ]  # This is the unique ticker (e.g., TSLA260116C00200000)

            if ask <= 0.05:
                continue  # Filter for basic liquidity

            # Determine Probability & ITM Status
            if row["type"] == "call":
                model_prob = np.mean(model_prices_at_t > k)
                is_itm = spot > k
            else:
                model_prob = np.mean(model_prices_at_t < k)
                is_itm = spot < k

            # Premium-based Exit Logic (Adjust multipliers as needed)
            tp_target, sl_target = calculate_distribution_exits(row, sim_df)

            all_results.append(
                {
                    "ticker": ticker_symbol,
                    "option_symbol": symbol,
                    "expiry": date_str,
                    "strike": k,
                    "type": row["type"],
                    "is_itm": is_itm,
                    "entry_range": f"${bid:.2f} - ${ask:.2f}",
                    "ask": ask,
                    "model_prob": model_prob,
                    "tp_target": tp_target,
                    "sl_target": sl_target,
                    "iv": row["impliedVolatility"],
                }
            )

    if not all_results:
        return None

    comparison_df = pd.DataFrame(all_results)

    # Calculate Kelly
    results = comparison_df.apply(
        lambda row: calculate_full_kelly_path_aware(row, sim_df), axis=1
    )
    comparison_df[["kelly_fraction", "expected_profit"]] = pd.DataFrame(
        results.tolist(), index=comparison_df.index
    )

    # Visualization and Saving
    save_kelly_charts(comparison_df, ticker_symbol)

    # 1. Add Metadata for History Tracking
    # This allows you to compare different versions of your 'Panelbeater' world model later
    comparison_df["run_timestamp"] = datetime.now()
    comparison_df["ticker"] = ticker_symbol

    # 2. Cleanup Data for Storage
    # We ensure decimals are kept as floats for mathematical precision in future reads
    export_df = comparison_df.copy()

    # 3. Export to Parquet
    # We use 'pyarrow' as the engine for better handling of complex types
    filename = f"panelbeater_signals_{ticker_symbol}.parquet"
    export_df.to_parquet(
        filename,
        engine="pyarrow",
        compression="snappy",  # High performance compression
        index=False,
    )

    print(f"📊 Analysis complete. Saved {len(export_df)} strikes to {filename}")
    return export_df


def save_kelly_charts(df, ticker):
    """Generates and saves separate Plotly charts for ITM and OTM options."""
    for status in [True, False]:
        label = "ITM" if status else "OTM"
        subset = df[(df["is_itm"] == status) & (df["kelly_fraction"] > 0)].copy()

        if subset.empty:
            continue

        fig = px.scatter(
            subset,
            x="strike",
            y="kelly_fraction",
            color="expiry",
            size="model_prob",
            symbol="type",
            # Include option_symbol in hover data for easy identification
            hover_data=["option_symbol", "entry_range", "tp_target", "sl_target"],
            title=f"{ticker} - {label} Kelly Conviction by Expiry",
            labels={
                "kelly_fraction": "Kelly Allocation (%)",
                "strike": "Strike Price ($)",
            },
            template="plotly_dark",
        )

        fig.update_layout(legend_title_text="Expiration Date")

        # Saving files
        png_path = f"kelly_{label}_{ticker}.png"
        fig.write_image(png_path, width=1400, height=800, scale=2)
        print(f"Chart saved: {png_path}")


def determine_spot_position_and_save(
    ticker_symbol: str, sim_df: pd.DataFrame
) -> pd.DataFrame:
    """Spot positions with dynamic boundary tightening and path-aware Kelly."""
    ticker = yf.Ticker(ticker_symbol)
    spot_price = ticker.history(period="1d")["Close"].iloc[-1]
    last_date = sim_df.index.max()
    date_str = last_date.strftime("%Y-%m-%d")  # pyright: ignore

    # Force path_matrix to be 2D (Rows: Time, Cols: Paths)
    # This prevents the IndexError if sim_df has unexpected dimensions
    path_matrix = np.atleast_2d(sim_df.values)

    # If sim_df was (N, 1), path_matrix is (N, 1).
    # If sim_df was just (N,), path_matrix becomes (1, N). We need (N, 1) or (N, M).
    if path_matrix.shape[0] == 1 and len(sim_df.index) > 1:
        path_matrix = path_matrix.T

    terminal_prices = path_matrix[-1]

    # 1. Determine Direction and Dynamic Boundaries
    terminal_std = np.std(terminal_prices)
    is_long = np.median(terminal_prices) > spot_price
    volatility_ratio = terminal_std / spot_price

    # Tighten percentiles if volatility is high
    if volatility_ratio > 0.15:
        tp_pct, sl_pct = (80, 20) if is_long else (20, 80)
    else:
        tp_pct, sl_pct = (95, 5) if is_long else (5, 95)

    tp_level = np.percentile(terminal_prices, tp_pct)
    sl_level = np.percentile(terminal_prices, sl_pct)

    # 2. Path-Aware Outcome Calculation
    path_outcomes = []

    # Iterate through each column (each simulation path)
    num_paths = path_matrix.shape[1]
    for col in range(num_paths):
        single_path = path_matrix[:, col]

        if is_long:
            hit_tp_idx = np.where(single_path >= tp_level)[0]
            hit_sl_idx = np.where(single_path <= sl_level)[0]
        else:
            hit_tp_idx = np.where(single_path <= tp_level)[0]
            hit_sl_idx = np.where(single_path >= sl_level)[0]

        # Determine which boundary was hit first
        first_tp = hit_tp_idx[0] if len(hit_tp_idx) > 0 else float("inf")
        first_sl = hit_sl_idx[0] if len(hit_sl_idx) > 0 else float("inf")

        if first_tp < first_sl:
            path_outcomes.append((tp_level - spot_price) / spot_price)
        elif first_sl < first_tp:
            path_outcomes.append((sl_level - spot_price) / spot_price)
        else:
            path_outcomes.append((single_path[-1] - spot_price) / spot_price)

    # 3. Mean-Variance Kelly
    path_returns = np.array(path_outcomes)
    actual_returns = path_returns if is_long else -path_returns

    mean_r = np.mean(actual_returns)
    var_r = np.var(actual_returns)

    # Pure Kelly: f* = E[r] / Var(r)
    if var_r > 0 and mean_r > 0:
        kelly_size = mean_r / var_r
    else:
        kelly_size = 0

    # 4. Final Metadata and Save
    spot_data = [
        {
            "run_timestamp": datetime.now(),
            "ticker": ticker_symbol,
            "option_symbol": f"{ticker_symbol}-SPOT",
            "expiry": date_str,
            "strike": spot_price,
            "type": "spot_long" if is_long else "spot_short",
            "is_itm": True,
            "entry_range": f"${spot_price:.2f}",
            "ask": spot_price,
            "model_prob": np.mean(
                actual_returns > 0
            ),  # Simplified prob based on path outcomes
            "tp_target": tp_level,
            "sl_target": sl_level,
            "iv": None,
            "kelly_fraction": max(0, kelly_size),
            "expected_profit": mean_r * 100,
            "volatility_regime": "High" if volatility_ratio > 0.15 else "Normal",
        }
    ]

    df = pd.DataFrame(spot_data)
    filename = f"panelbeater_spot_{ticker_symbol}.parquet"
    df.to_parquet(filename, engine="pyarrow", compression="snappy", index=False)

    print(f"✅ Path-aware spot saved. Paths: {num_paths}, Kelly: {kelly_size:.2%}")
    return df
