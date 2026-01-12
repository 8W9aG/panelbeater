"""Handle generating trades."""

# pylint: disable=use-dict-literal
import pandas as pd
import tqdm

from .options import determine_spot_position, find_mispriced_options
from .simulate import SIMULATION_COLUMN, load_simulations


def trades(df_y: pd.DataFrame, days_out: int, tickers: list[str]) -> None:
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
        # Add any additional styling
        fig.add_scatter(
            x=plot_df.index,
            y=plot_df.median(axis=1),
            name="Median",
            line=dict(color="white", width=10),
        )
        fig.write_image(
            f"monte_carlo_results_{col}.png", width=1200, height=800, scale=2
        )

    # Find the current options prices
    for ticker in tickers:
        print(f"Finding pricing options for {ticker}")
        find_mispriced_options(ticker, df_mc[f"PX_{ticker}"].copy())  # pyright: ignore
        determine_spot_position(ticker, df_mc[f"PX_{ticker}"].copy())  # pyright: ignore
