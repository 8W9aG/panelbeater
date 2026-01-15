"""Handle generating trades."""

# pylint: disable=use-dict-literal,line-too-long
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
    all_trades = []
    for ticker in tickers:
        print(f"Finding pricing options for {ticker}")
        options_trades = find_mispriced_options_comprehensive(
            ticker,
            df_mc[f"PX_{ticker}"].copy(),  # pyright: ignore
        )  # pyright: ignore
        if options_trades is not None:
            all_trades.append(options_trades)
        spot_trades = determine_spot_position_and_save(
            ticker,
            df_mc[f"PX_{ticker}"].copy(),  # pyright: ignore
        )  # pyright: ignore
        all_trades.append(spot_trades)
    return pd.concat(all_trades, axis=0, ignore_index=True)
