"""Process the options for the assets."""

# pylint: disable=too-many-locals,consider-using-f-string,use-dict-literal,invalid-name,too-many-arguments,too-many-positional-arguments,too-many-statements,line-too-long,bare-except,too-many-branches
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf

from .kelly import calculate_full_kelly_path_aware
from .sizing import (apply_merton_jumps, calculate_distribution_exits,
                     calculate_path_aware_mean_variance, prepare_path_matrix)


def calculate_model_volatility(wide_df: pd.DataFrame, days_per_year=365) -> float:
    """Calculates annualized volatility from path matrix."""
    # Standard deviation of log returns across all paths
    log_rets = np.log(wide_df / wide_df.shift(1)).dropna()
    daily_vol = log_rets.std().mean()  # Average std across all simulation columns
    return daily_vol * np.sqrt(days_per_year)


def apply_volatility_sanity_filter(row, model_vol, benchmark_vol_placeholder=0.18):
    """Applies a volatility sanity filter."""
    market_iv = row["impliedVolatility"]

    # Check 1: Vol vs Market IV
    if model_vol > (market_iv * 1.5):
        return False, f"VOL_TOO_HIGH: {model_vol:.2%} vs IV {market_iv:.2%}"

    # Check 2: QQQ/SPY Ratio
    ratio = model_vol / benchmark_vol_placeholder
    if ratio > 2.0:
        return False, f"RATIO_TOO_HIGH: {ratio:.2f}x benchmark"

    return True, "PASS"


def find_mispriced_options_comprehensive(
    ticker_symbol: str, sim_df: pd.DataFrame
) -> pd.DataFrame | None:
    """Comprehensively find mispriced options in ITM and OTM."""
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="1d")

    if history.empty:
        print(f"Warning: No data found for {ticker.ticker}. Skipping.")
        return None

    spot = history["Close"].iloc[-1]
    sim_dates = pd.to_datetime(sim_df.index).date.tolist()  # pyright: ignore
    available_expiries = [
        datetime.strptime(d, "%Y-%m-%d").date() for d in ticker.options
    ]
    common_dates = sorted(list(set(sim_dates).intersection(set(available_expiries))))

    all_results = []

    # Counters for liquidity filtering logging
    total_chains_seen = 0
    dropped_liquidity = 0

    for target_date in common_dates:
        date_str = target_date.strftime("%Y-%m-%d")
        chain = ticker.option_chain(date_str)

        calls = chain.calls.copy()
        puts = chain.puts.copy()
        calls["type"] = "call"
        puts["type"] = "put"
        calls["expiry"] = date_str
        puts["expiry"] = date_str

        full_chain = pd.concat([calls, puts])

        # Track initial count
        initial_len = len(full_chain)
        total_chains_seen += initial_len

        # --- LIQUIDITY FILTER START (SMART v2) ---
        # 1. Basic Volume/Interest Check
        # We allow 0 volume if OI is high (e.g. morning trading)
        full_chain = full_chain[
            (full_chain["openInterest"] > 10)
            & (full_chain["volume"] >= 0)
            & (full_chain["ask"] > 0.05)
        ].copy()

        # 2. Calculate Spreads
        full_chain["spread_width"] = full_chain["ask"] - full_chain["bid"]
        full_chain["rel_spread"] = full_chain["spread_width"] / full_chain["ask"]

        # 3. Smart Filter Logic
        # Rule A: Percentage spread is healthy (<= 20%)
        # Rule B: Dollar spread is tiny (<= $0.15), regardless of percentage.
        #         (This saves cheap OTM options like Bid 0.20 / Ask 0.25)
        liquidity_mask = (full_chain["rel_spread"] <= 0.20) | (
            full_chain["spread_width"] <= 0.15
        )

        full_chain = full_chain[liquidity_mask].copy()

        # 4. Realistic Entry Price
        full_chain["effective_entry"] = full_chain["ask"]

        dropped_liquidity += initial_len - len(full_chain)
        # --- LIQUIDITY FILTER END ---

        model_prices_at_t = sim_df.loc[date_str].values

        for _, row in full_chain.iterrows():  # pyright: ignore
            k = row["strike"]
            ask = row["ask"]
            bid = row["bid"]
            symbol = row["contractSymbol"]

            if ask <= 0.05:
                continue

            # Determine Probability & ITM Status
            if row["type"] == "call":
                model_prob = np.mean(model_prices_at_t > k)
                is_itm = spot > k
            else:
                model_prob = np.mean(model_prices_at_t < k)
                is_itm = spot < k

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
                    "ask": (bid + ask) / 2 * 1.02,
                    "model_prob": model_prob,
                    "tp_target": tp_target,
                    "sl_target": sl_target,
                    "impliedVolatility": row["impliedVolatility"],
                }
            )

    # Log the liquidity drops
    print(
        f"📉 Liquidity Filter: Dropped {dropped_liquidity}/{total_chains_seen} contracts (Low Vol/OI/High Spread)."
    )

    if not all_results:
        print("❌ No contracts survived the basic liquidity filter.")
        return None

    comparison_df = pd.DataFrame(all_results)

    # --- VOLATILITY SANITY FILTER START ---
    wide_sim_df = prepare_path_matrix(sim_df, ticker_symbol)
    model_vol = calculate_model_volatility(wide_sim_df, days_per_year=365)

    # Apply logic but DO NOT DROP ROWS yet
    # We apply the filter and store the reason
    def _apply_filter_wrapper(row):
        return apply_volatility_sanity_filter(
            row, model_vol, benchmark_vol_placeholder=0.18
        )

    # Apply returns a tuple (bool, reason), so we unzip it
    filter_results = comparison_df.apply(_apply_filter_wrapper, axis=1)
    comparison_df["is_rational"] = [res[0] for res in filter_results]
    comparison_df["filter_reason"] = [res[1] for res in filter_results]
    comparison_df["model_vol_annualized"] = model_vol

    # LOGGING: Detailed breakdown of reasons
    print("\n🔍 Filter Reason Breakdown:")
    print(comparison_df["filter_reason"].value_counts().to_string())
    # --- VOLATILITY SANITY FILTER END ---

    # Apply the Black Swan "Truth Serum"
    path_values = apply_merton_jumps(
        wide_sim_df.values, days_per_year=365, lam=1.0, mu_j=-0.15
    )
    wide_sim_df = pd.DataFrame(
        path_values, index=wide_sim_df.index, columns=wide_sim_df.columns
    )

    # --- CONDITIONAL KELLY CALCULATION ---
    # We now calculate Kelly ONLY if the trade is rational.
    # If not rational, we force 0.0. This ensures the row exists in the dataframe
    # so the sync script can see "Kelly: 0.0" and close the position.

    def _safe_kelly(row):
        if not row["is_rational"]:
            return 0.0, 0.0  # Force zero conviction for filtered trades
        return calculate_full_kelly_path_aware(row, wide_sim_df)

    results = comparison_df.apply(_safe_kelly, axis=1)

    comparison_df[["kelly_fraction", "expected_profit"]] = pd.DataFrame(
        results.tolist(), index=comparison_df.index
    )

    # Visualization and Saving (Only graph the positive ones to avoid clutter)
    save_kelly_charts(comparison_df, ticker_symbol)

    comparison_df["run_timestamp"] = datetime.now()
    comparison_df["ticker"] = ticker_symbol

    export_df = comparison_df.copy()
    filename = f"panelbeater_signals_{ticker_symbol}.parquet"

    # Save EVERYTHING, including the 0.0 conviction trades
    export_df.to_parquet(
        filename,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    valid_trades = len(export_df[export_df["kelly_fraction"] > 0])
    print(
        f"📊 Analysis complete. Saved {len(export_df)} rows ({valid_trades} actionable) to {filename}"
    )

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
) -> pd.DataFrame | None:
    """Determine spot position."""
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1d")
    if hist.empty:
        print(f"Skipping {ticker.ticker}: No price data found.")
        return None  # Or handle strictly as needed

    spot_price = hist["Close"].iloc[-1]

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
