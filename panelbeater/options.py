"""Process the options for the assets."""

# pylint: disable=too-many-locals,consider-using-f-string,use-dict-literal,invalid-name,too-many-arguments,too-many-positional-arguments,too-many-statements,line-too-long
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf
from scipy import stats


def get_price_probabilities(sim_df, target_date, bin_width=1.0):
    """
    Calculates the probability distribution of prices for a specific date.

    Args:
        sim_df: The simulation DataFrame (rows=dates, cols=paths)
        target_date: The specific date (index) or integer location to analyze
        bin_width: The size of the price buckets (e.g., $1.00)
    """
    # 1. Slice the simulation at the specific point in time
    # This handles both a date-string index or a simple integer row index
    if isinstance(target_date, int):
        prices_at_t = sim_df.iloc[target_date]
    else:
        prices_at_t = sim_df.loc[target_date]

    # 2. Define bins based on the range of prices on that specific day
    min_p = np.floor(prices_at_t.min() / bin_width) * bin_width
    max_p = np.ceil(prices_at_t.max() / bin_width) * bin_width
    bins = np.arange(min_p, max_p + bin_width, bin_width)

    # 3. Calculate probabilities
    counts, bin_edges = np.histogram(prices_at_t, bins=bins)
    probabilities = counts / len(prices_at_t)

    # 4. Format into a DataFrame
    price_points = bin_edges[:-1] + (bin_width / 2)
    dist_df = pd.DataFrame({"price_point": price_points, "probability": probabilities})

    return dist_df[dist_df["probability"] > 0].reset_index(drop=True)


def calculate_full_kelly(row, sim_df):
    """Calculate the kelly criterion for a probability mispricing."""
    target_date = row["date"]
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


def black_scholes_price(S, K, T, r, sigma, option_type="put"):
    """Calculate the black scholes price for an option."""
    # S = Trigger Asset Price, K = Strike, T = Time remaining, r = Risk-free rate, sigma = IV
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def find_mispriced_options(ticker_symbol: str, sim_df: pd.DataFrame) -> None:
    """Find any mispriced options for an asset."""

    # 1. Initialize the Ticker
    ticker = yf.Ticker(ticker_symbol)

    # 1. Get dates from your simulation
    sim_dates = pd.to_datetime(sim_df.index).date.tolist()  # pyright: ignore

    # 2. Get available expiries from the market
    available_expiries = [
        datetime.strptime(d, "%Y-%m-%d").date() for d in ticker.options
    ]

    # 3. Find the common dates
    # We want to find which days in our simulation actually have a tradeable option chain
    common_dates = sorted(list(set(sim_dates).intersection(set(available_expiries))))

    print(f"Simulation covers {len(sim_dates)} days.")
    print(f"Market has {len(available_expiries)} expiries available.")
    print(f"Matches found for: {common_dates}")

    # Storage for our comparison results
    date_results = []

    for target_date in common_dates:
        print(f"\n--- Processing Date: {target_date} ---")

        # 1. Get YOUR model's probability for this specific day
        # We use the function we built earlier
        date_str = target_date.strftime("%Y-%m-%d")

        # 2. Download the MARKET's chain for this specific day
        chain = ticker.option_chain(date_str)
        spot = ticker.history(period="1d")["Close"].iloc[-1]
        calls = chain.calls[["strike", "bid", "ask", "impliedVolatility"]].copy()
        calls = calls[calls["strike"] > spot * 1.02]
        calls["option_type"] = "call"
        puts = chain.puts[["strike", "bid", "ask", "impliedVolatility"]].copy()
        puts = puts[puts["strike"] < spot * 0.98]
        puts["option_type"] = "put"

        # 3. Combine into one market view
        full_chain = pd.concat([calls, puts])

        # 4. Get your Model's Price Distribution for this specific day
        # We grab the prices from sim_df for this row/date
        model_prices_at_t = sim_df.loc[date_str].values

        # 5. Compare every strike in the market to your model's probability
        for _, row in full_chain.iterrows():
            k = row["strike"]

            if row["option_type"] == "call":
                # Prob of finishing ABOVE the strike
                model_prob = np.mean(model_prices_at_t > k)
            else:
                # Prob of finishing BELOW the strike
                model_prob = np.mean(model_prices_at_t < k)

            date_results.append(
                {
                    "date": date_str,
                    "strike": k,
                    "type": row["option_type"],
                    "market_iv": row["impliedVolatility"],
                    "market_ask": row["ask"],
                    "model_prob": model_prob,
                }
            )

    comparison_df = pd.DataFrame(date_results)
    # Apply the calculation
    results = comparison_df.apply(lambda row: calculate_full_kelly(row, sim_df), axis=1)
    comparison_df[["kelly_fraction", "expected_profit"]] = pd.DataFrame(
        results.tolist(), index=comparison_df.index
    )

    # Filter for liquid options and positive edge
    top_5 = (
        comparison_df[comparison_df["market_ask"] > 0.10]  # pyright: ignore
        .sort_values(by="kelly_fraction", ascending=False)
        .head(4)
    )

    # Formatting for the final report
    summary_report = top_5[
        ["date", "strike", "type", "model_prob", "kelly_fraction", "expected_profit"]
    ].copy()
    summary_report["model_prob"] = summary_report["model_prob"].map("{:.1%}".format)  # pyright: ignore
    summary_report["kelly_fraction"] = summary_report["kelly_fraction"].map(  # pyright: ignore
        "{:.2%}".format
    )
    summary_report["expected_profit"] = summary_report["expected_profit"].map(  # pyright: ignore
        "${:,.2f}".format
    )

    print(summary_report)

    fig = px.scatter(
        comparison_df[comparison_df["kelly_fraction"] > 0],
        x="strike",
        y="kelly_fraction",
        color="type",
        size="model_prob",
        hover_data=["date"],
        title="Full Kelly Allocation: Conviction by Strike and Option Type",
        labels={"kelly_fraction": "Kelly Bet Size (%)", "strike": "Strike Price ($)"},
        template="plotly_dark",
    )

    # Highlight the top 5 with annotations or larger markers
    fig.update_traces(marker=dict(line=dict(width=1, color="White")))
    fig.write_image(
        f"kelly_conviction_report_{ticker_symbol}.png", width=1200, height=800
    )

    exit_strategies = []

    for _, trade in top_5.iterrows():
        # Select appropriate simulation slices
        sim_slice = sim_df.loc[trade["date"]]

        # --- ADJUSTED LOGIC START ---
        if trade["type"] == "call":
            # Call: Profit on upside (95th percentile), Stop on downside (5th percentile)
            tp_price = sim_slice.quantile(0.95)
            sl_price = sim_slice.quantile(0.05)
        else:
            # Put: Profit on downside (5th percentile), Stop on upside (95th percentile)
            tp_price = sim_slice.quantile(0.05)
            sl_price = sim_slice.quantile(0.95)
        # --- ADJUSTED LOGIC END ---

        # 1. Get today's date and calculate time to expiry
        today = datetime.now()
        expiry_date = datetime.strptime(trade["date"], "%Y-%m-%d")  # type: ignore
        days_remaining = (expiry_date - today).days
        time_to_expiry = max(days_remaining, 0.5) / 365.0

        # Calculate the Black-Scholes prices for the options at these triggers
        tp_option_price = black_scholes_price(
            tp_price,
            trade["strike"],
            time_to_expiry,
            0.04,
            trade["market_iv"],
            str(trade["type"]),
        )
        sl_option_price = black_scholes_price(
            sl_price,
            trade["strike"],
            time_to_expiry,
            0.04,
            trade["market_iv"],
            str(trade["type"]),
        )

        exit_strategies.append(
            {
                "Strike": trade["strike"],
                "Type": trade["type"],
                "Kelly %": trade["kelly_fraction"],
                "TP Asset Trigger": tp_price,
                "SL Asset Trigger": sl_price,
                "TP Option Price": tp_option_price,
                "SL Option Price": sl_option_price,
            }
        )

    exit_df = pd.DataFrame(exit_strategies)
    print(exit_df)
