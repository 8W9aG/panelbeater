"""The CLI for finding mispriced options."""

# pylint: disable=too-many-locals,use-dict-literal,invalid-name
import argparse

import pandas as pd
import requests_cache
import tqdm
from dotenv import load_dotenv
from joblib import Parallel, delayed

from .copula import load_vine_copula, sample_joint_step
from .download import download
from .features import features
from .fit import fit
from .normalizer import denormalize
from .options import determine_spot_position, find_mispriced_options
from .wt import create_wt

_TICKERS = [
    # Equities
    "SPY",
    "QQQ",
    "EEM",
    # Commodities
    "GC=F",
    "CL=F",
    "SI=F",
    # FX
    # "EURUSD=X",
    # "USDJPY=X",
    # Crypto
    # "BTC-USD",
    # "ETH-USD",
]
_MACROS = [
    "GDP",
    "UNRATE",
    "CPIAUCSL",
    "FEDFUNDS",
    "DGS10",
    "T10Y2Y",
    # "M2SL",
    # "VIXCLS",
    # "DTWEXBGS",
    # "INDPRO",
]
_WINDOWS = [
    5,
    10,
    20,
    60,
    120,
    200,
]
_LAGS = [1, 3, 5, 10, 20, 30]
_DAYS_OUT = 30
_SIMS = 1000
_SIMULATION_COLUMN = "simulation"


def run_single_simulation(sim_idx, df_y, _DAYS_OUT, _WINDOWS, _LAGS):
    """
    Encapsulates a single Monte Carlo path generation.
    """
    # Local copies for thread-safety (though joblib uses processes)
    df_y = df_y.copy()
    vine_cop = load_vine_copula(df_returns=df_y)
    wavetrainer = create_wt()

    for _ in range(_DAYS_OUT):
        # 1. Feature Engineering
        df_x = features(df=df_y.copy(), windows=_WINDOWS, lags=_LAGS)

        # 2. Get Model Prediction (u_step sample from Copula)
        u_step = sample_joint_step(vine_cop)

        # 3. Transform and Denormalize to get next day prices
        df_next = wavetrainer.transform(df_x.iloc[[-1]], ignore_no_dates=True).drop(
            columns=df_x.columns.values.tolist()
        )
        df_y = denormalize(df_next, y=df_y.copy(), u_sample=u_step)

    # Mark the simulation index and return only the relevant tail (for memory efficiency)
    df_result = df_y.tail(_DAYS_OUT + 1).copy()
    df_result[_SIMULATION_COLUMN] = sim_idx
    return df_result


def main() -> None:
    """The main CLI function."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference",
        help="Whether to do inference.",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--train",
        help="Whether to do training.",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    # Setup main objects
    session = requests_cache.CachedSession("panelbeater-cache")

    # Fit the models
    df_y = download(tickers=_TICKERS, macros=_MACROS, session=session)
    if args.train:
        fit(df_y=df_y, windows=_WINDOWS, lags=_LAGS)

    if args.inference:
        print(f"Starting {_SIMS} simulations in parallel...")

        # n_jobs=-1 uses all available CPU cores
        all_sims = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(i, df_y.copy(), _DAYS_OUT, _WINDOWS, _LAGS)
            for i in tqdm.tqdm(range(_SIMS), desc="Simulating")
        )

        # Combine all simulations into one large DataFrame
        df_mc = pd.concat(all_sims)  # type: ignore
        pd.options.plotting.backend = "plotly"
        for col in tqdm.tqdm(df_y.columns.values.tolist(), desc="Plotting assets"):
            if col == _SIMULATION_COLUMN:
                continue
            plot_df = df_mc.pivot(columns=_SIMULATION_COLUMN, values=col).tail(
                _DAYS_OUT + 1
            )
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
        for ticker in _TICKERS:
            print(f"Finding pricing options for {ticker}")
            find_mispriced_options(ticker, df_mc[f"PX_{ticker}"])  # pyright: ignore
            determine_spot_position(ticker, df_mc[f"PX_{ticker}"])  # pyright: ignore


if __name__ == "__main__":
    main()
