"""Process the options for the assets."""

# pylint: disable=too-many-locals,consider-using-f-string,use-dict-literal,invalid-name,too-many-arguments,too-many-positional-arguments,too-many-statements,line-too-long,bare-except,too-many-branches
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf

from .kelly import calculate_full_kelly_path_aware
from .sizing import (calculate_distribution_exits,
                     calculate_path_aware_mean_variance, prepare_path_matrix)


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
                    "impliedVolatility": row["impliedVolatility"],
                }
            )

    if not all_results:
        return None

    comparison_df = pd.DataFrame(all_results)

    # Calculate Kelly
    wide_sim_df = prepare_path_matrix(sim_df, ticker_symbol)
    print(f"DEBUG: Wide Matrix Shape: {wide_sim_df.shape}")
    results = comparison_df.apply(
        lambda row: calculate_full_kelly_path_aware(row, wide_sim_df), axis=1
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
    """Determine spot position."""
    ticker = yf.Ticker(ticker_symbol)
    spot_price = ticker.history(period="1d")["Close"].iloc[-1]

    # 1. Transform Long -> Wide
    # This turns your stacked simulations into a 2D Path Matrix
    wide_paths = prepare_path_matrix(sim_df, ticker_symbol)

    path_matrix = wide_paths.values  # Shape: (TimeSteps, NumSimulations)
    terminal_prices = path_matrix[-1]

    # 2. Dynamic Boundaries
    terminal_std = np.std(terminal_prices)
    is_long = np.median(terminal_prices) > spot_price
    volatility_ratio = terminal_std / spot_price

    # Tighten targets if volatility is high
    tp_pct, sl_pct = (
        ((80, 20) if is_long else (20, 80))
        if volatility_ratio > 0.15
        else ((95, 5) if is_long else (5, 95))
    )

    tp_level = np.percentile(terminal_prices, tp_pct)
    sl_level = np.percentile(terminal_prices, sl_pct)

    mean_r, _, kelly_size, actual_returns = calculate_path_aware_mean_variance(
        path_matrix, spot_price, is_long, tp_level, sl_level
    )

    # 4. Final Metadata and Save
    spot_data = [
        {
            "run_timestamp": datetime.now(),
            "ticker": ticker_symbol,
            "option_symbol": f"{ticker_symbol}-SPOT",
            "expiry": None,
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

    print(
        f"✅ Path-aware spot saved. Paths: {len(actual_returns)}, Kelly: {kelly_size:.2%}"
    )
    return df
