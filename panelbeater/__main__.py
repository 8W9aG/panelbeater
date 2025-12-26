"""The CLI for finding mispriced options."""

# pylint: disable=too-many-locals,use-dict-literal
import argparse
import datetime

import pandas as pd
import requests_cache
import tqdm
import wavetrainer as wt
from dotenv import load_dotenv

from .copula import fit_vine_copula, sample_joint_step
from .download import download
from .features import features
from .normalizer import denormalize, normalize
from .options import find_mispriced_options

_TICKERS = [
    # Equities
    "SPY",
    "QQQ",
    "EEM",
    # Commodities
    "GC=F",
    # "CL=F",
    # "SI=F",
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
    # "DGS10",
    # "T10Y2Y",
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


def main() -> None:
    """The main CLI function."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference",
        help="Whether to skip training and just do inference.",
        required=False,
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    # Setup main objects
    session = requests_cache.CachedSession("panelbeater-cache")
    wavetrainer = wt.create(
        "panelbeater-train",
        walkforward_timedelta=datetime.timedelta(days=30),
        validation_size=datetime.timedelta(days=365),
        test_size=datetime.timedelta(days=365),
        allowed_models={"catboost"},
        max_false_positive_reduction_steps=0,
    )

    # Fit the models
    df_y = download(tickers=_TICKERS, macros=_MACROS, session=session)

    # Fit Vine Copula on historical returns
    # We use pct_change to capture the dependency of returns
    returns = df_y.pct_change().dropna()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    vine_cop = fit_vine_copula(returns)

    # Simulate the paths
    all_sims = []
    original_df_y = df_y.copy()
    for sim_idx in tqdm.tqdm(range(_SIMS), desc="Simulations"):
        u_step = sample_joint_step(vine_cop)
        df_x = features(df=original_df_y.copy(), windows=_WINDOWS, lags=_LAGS)
        df_y_norm = normalize(df=original_df_y.copy())
        if not args.inference and sim_idx == 0:
            wavetrainer.fit(df_x, y=df_y_norm)
        df_y = original_df_y.copy()
        for _ in tqdm.tqdm(range(_DAYS_OUT), desc="Running t+X simulation"):
            df_next = wavetrainer.transform(df_x.iloc[[-1]], ignore_no_dates=True).drop(
                columns=df_x.columns.values.tolist()
            )
            df_y = denormalize(df_next, y=df_y.copy(), u_sample=u_step)
            df_x = features(df=df_y.copy(), windows=_WINDOWS, lags=_LAGS)
            df_y_norm = normalize(df=df_y.copy())
        df_y[_SIMULATION_COLUMN] = sim_idx
        print(df_y.tail(_DAYS_OUT + 1))
        all_sims.append(df_y.copy())

    # Combine all simulations into one large DataFrame
    df_mc = pd.concat(all_sims)
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
            line=dict(color="white", width=3),
        )
        fig.write_image(
            f"monte_carlo_results_{col}.png", width=1200, height=800, scale=2
        )

    # Find the current options prices
    for ticker in _TICKERS:
        print(f"Finding pricing options for {ticker}")
        find_mispriced_options(ticker, df_mc[f"PX_{ticker}"])  # pyright: ignore


if __name__ == "__main__":
    main()
