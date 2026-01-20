"""Handle generating trades."""

# pylint: disable=use-dict-literal,line-too-long,no-else-return
import pandas as pd
import tqdm

from .options import (determine_spot_position_and_save,
                      find_mispriced_options_comprehensive)
from .simulate import SIMULATION_COLUMN, load_simulations


def trades(df_y: pd.DataFrame, days_out: int, tickers: list[str]) -> pd.DataFrame:
    """Calculate new trades."""
    df_mc = load_simulations()
    pd.options.plotting.backend = "plotly"
    for col in tqdm.tqdm(df_y.columns.values.tolist(), desc="Plotting assets"):
        if col == SIMULATION_COLUMN:
            continue
        plot_df = df_mc.pivot(columns=SIMULATION_COLUMN, values=col).tail(days_out + 1)
        # Plotting
        fig = plot_df.plot(
            title=f"Monte Carlo Simulation: {col}",
            labels={"value": "Price", "index": "Date", "simulation": "Path ID"},
            template="plotly_dark",
        )

        # 2. KEY FIX: Dim the simulation lines immediately
        # This makes them semi-transparent (opacity) and thin, so they form a
        # background "cloud" rather than a solid wall of color.
        fig.update_traces(
            line=dict(width=1),
            opacity=0.3,  # Adjust between 0.1 and 0.5 depending on number of sims
        )

        # Add any additional styling
        fig.add_scatter(
            x=plot_df.index,
            y=plot_df.median(axis=1),
            name="Median",
            line=dict(
                color="white", width=8
            ),  # Slightly reduced width often looks sharper
            opacity=1.0,  # Ensure median is fully opaque
        )

        fig.write_image(
            f"monte_carlo_results_{col}.png", width=1200, height=800, scale=2
        )

    # Find the current options prices
    # Find the current options prices
    all_trades = []
    for ticker in tickers:
        print(f"Finding pricing options for {ticker}")

        # 1. Define the columns we need to keep
        price_col = f"PX_{ticker}"
        required_cols = [price_col, SIMULATION_COLUMN]

        # 2. Filter the MC dataframe but keep it as a DataFrame
        # This ensures 'simulation' metadata travels with the prices
        ticker_sim_data = df_mc[required_cols].copy()

        options_trades = find_mispriced_options_comprehensive(
            ticker,
            ticker_sim_data,  # pyright: ignore
        )
        if options_trades is not None:
            all_trades.append(options_trades)

        spot_trades = determine_spot_position_and_save(
            ticker,
            ticker_sim_data,  # pyright: ignore
        )
        all_trades.append(spot_trades)
    return pd.concat(all_trades, axis=0, ignore_index=True)


def process_and_classify_trades(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the raw analysis into 'Actionable Opportunities' (Buy)
    and 'Management Signals' (Hold/Sell).

    Returns:
        opportunities_df: Only trades with Positive EV and Kelly > 0
        portfolio_view_df: All trades with added 'Recommendation' columns
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    processed_df = df.copy()

    # 1. Calculate Expected Edge
    # If this is negative, the market price is higher than our model's Take Profit
    processed_df["upside_edge"] = (
        processed_df["tp_target"] - processed_df["ask"]
    ) / processed_df["ask"]

    # 2. Assign Recommendations
    def assign_action(row):
        if row["kelly_fraction"] > 0 and row["upside_edge"] > 0:
            return "BUY_NEW"
        if row["upside_edge"] < 0:
            return "OVERVALUED_SELL"  # Price is above our model's take profit
        if row["kelly_fraction"] == 0:
            return "AVOID_NO_EDGE"
        return "HOLD"

    processed_df["action"] = processed_df.apply(assign_action, axis=1)

    # 3. Create the "New Money" DataFrame
    # This is what you look at to open NEW trades
    opportunities_df = processed_df[
        (processed_df["action"] == "BUY_NEW")
        & (processed_df["model_prob"] > 0.4)  # Optional: Safety filter
    ].copy()

    # Sort opportunities by Kelly conviction
    opportunities_df = opportunities_df.sort_values(  # pyright: ignore
        by="kelly_fraction", ascending=False
    )

    # 4. The Portfolio View (Keep everything, just organized)
    portfolio_view_df = processed_df[  # pyright: ignore
        [
            "ticker",
            "option_symbol",
            "expiry",
            "strike",
            "type",
            "ask",
            "tp_target",
            "sl_target",
            "upside_edge",
            "kelly_fraction",
            "action",
        ]
    ].sort_values(by=["expiry", "strike"])

    return opportunities_df, portfolio_view_df
