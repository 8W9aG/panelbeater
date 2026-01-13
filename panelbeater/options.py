"""Process the options for the assets."""

# pylint: disable=too-many-locals,consider-using-f-string,use-dict-literal,invalid-name,too-many-arguments,too-many-positional-arguments,too-many-statements,line-too-long,bare-except
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
from scipy.stats import norm


def calculate_full_kelly(row, sim_df):
    """Calculate the kelly criterion for a probability mispricing."""
    target_date = row["expiry"]
    strike = row["strike"]
    price = row["market_ask"]

    if price <= 0:
        return 0, 0

    # Extract the simulated prices for this specific date
    prices_at_t = sim_df.loc[target_date].values

    # Calculate the Payoff for every path
    if row["type"] == "call":
        payoffs = np.maximum(prices_at_t - strike, 0)
    else:
        payoffs = np.maximum(strike - prices_at_t, 0)

    expected_payoff = np.mean(payoffs)

    # 1. Probability of winning (p)
    p = row["model_prob"]
    if p <= 0:
        return 0, 0

    # 2. Net Odds (b)
    # This is (Expected Profit if we win) / (Amount Lost if we lose)
    # Average payoff of the winning paths
    avg_win_payoff = expected_payoff / p
    net_profit_if_win = avg_win_payoff - price
    b = net_profit_if_win / price

    if b <= 0:
        return 0, 0

    # 3. Full Kelly Formula: f* = (p(b+1) - 1) / b
    f_star = (p * (b + 1) - 1) / b

    return max(0, f_star), expected_payoff - price


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
    Calculates TP/SL based on the percentile of predicted OPTION prices
    at a specific point in time (horizon_pct).
    """
    # Access the injected expiry column
    date_val = row["expiry"]

    # Ensure matching index types (sim_df index vs date_val string)
    if date_val not in sim_df.index:
        # If sim_df index is Datetime objects, convert date_val
        try:
            target_lookup = pd.to_datetime(date_val)
            sim_prices = sim_df.loc[target_lookup].values
        except KeyError:
            # Fallback: convert index to string if needed
            sim_prices = sim_df.loc[str(date_val)].values
    else:
        sim_prices = sim_df.loc[date_val].values

    # 2. Define the 'Check-in' time (Time to Expiry at our horizon)
    today = datetime.now()
    expiry_date = datetime.strptime(row["expiry"], "%Y-%m-%d")
    total_days = (expiry_date - today).days

    if total_days <= 0:
        return row["ask"], row["ask"]

    # We evaluate the option's value when 'horizon_pct' of time has passed
    # e.g., if 30 days left, we look at its value in 15 days
    days_to_horizon = total_days * horizon_pct
    time_to_expiry_at_horizon = (total_days - days_to_horizon) / 365.0

    # 3. Vectorized Black-Scholes: Map underlying distribution -> option distribution
    # This simulates what the OPTION will be worth across all paths
    predicted_option_values = black_scholes_price(
        sim_prices,
        row["strike"],
        time_to_expiry_at_horizon,
        0.04,  # Risk-free rate
        row["impliedVolatility"],
        row["type"],
    )

    # 4. Set exits based on the model's specific distribution
    # TP = 80th percentile (High confidence target)
    # SL = 15th percentile (Cutting before total wipeout)
    tp = np.percentile(predicted_option_values, 80)
    sl = np.percentile(predicted_option_values, 15)

    return tp, sl


def find_mispriced_options_comprehensive(
    ticker_symbol: str, sim_df: pd.DataFrame
) -> pd.DataFrame:
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

    comparison_df = pd.DataFrame(all_results)

    # Calculate Kelly
    results = comparison_df.apply(lambda row: calculate_full_kelly(row, sim_df), axis=1)
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
    """Determine the spot positions."""
    ticker = yf.Ticker(ticker_symbol)
    spot_price = ticker.history(period="1d")["Close"].iloc[-1]
    last_date = sim_df.index.max()
    date_str = last_date.strftime("%Y-%m-%d")  # pyright: ignore

    # Extract distribution
    terminal_prices = sim_df.loc[last_date].values
    median_terminal = np.median(terminal_prices)
    is_long = median_terminal > spot_price

    # 1. Distribution-Based Logic (Consistent with your Options logic)
    if is_long:
        p = np.mean(terminal_prices > spot_price)
        tp_price = np.percentile(terminal_prices, 95)
        sl_price = np.percentile(terminal_prices, 5)
    else:
        p = np.mean(terminal_prices < spot_price)
        tp_price = np.percentile(terminal_prices, 5)
        sl_price = np.percentile(terminal_prices, 95)

    # 2. Kelly Calculation
    expected_profit = abs(tp_price - spot_price)
    expected_loss = abs(spot_price - sl_price)
    b = expected_profit / expected_loss
    kelly_size = max(0, (p * (b + 1) - 1) / b) if b > 0 else 0

    # 3. Create Row with Unified Schema
    spot_data = [
        {
            "run_timestamp": datetime.now(),
            "ticker": ticker_symbol,
            "option_symbol": f"{ticker_symbol}-SPOT",  # Identifying as spot
            "expiry": date_str,
            "strike": spot_price,  # Entry as strike for spot
            "type": "spot_long" if is_long else "spot_short",
            "is_itm": True,  # Spot is always "at the money" effectively
            "entry_range": f"${spot_price:.2f}",
            "ask": spot_price,
            "model_prob": p,
            "tp_target": tp_price,
            "sl_target": sl_price,
            "iv": sim_df.pct_change().std().iloc[0]  # pyright: ignore
            * np.sqrt(252),  # Realized Vol Proxy
            "kelly_fraction": kelly_size,
            "expected_profit": (p * expected_profit) - ((1 - p) * expected_loss),
        }
    ]

    df = pd.DataFrame(spot_data)

    # 4. Save to Parquet
    filename = f"panelbeater_spot_{ticker_symbol}.parquet"
    df.to_parquet(filename, engine="pyarrow", compression="snappy", index=False)

    print(f"✅ Spot analysis saved to {filename}")
    return df
