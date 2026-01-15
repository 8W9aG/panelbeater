"""The CLI for finding mispriced options."""

# pylint: disable=too-many-locals,use-dict-literal,invalid-name
import argparse

import requests_cache
from dotenv import load_dotenv

from .download import download
from .fit import fit
from .simulate import simulate
from .sync import sync_positions
from .trades import trades

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
    parser.add_argument(
        "--trades",
        help="Whether to generate trades.",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--sync",
        help="Whether to synchronise the trades.",
        required=False,
        default=False,
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
        simulate(
            sims=_SIMS, df_y=df_y, days_out=_DAYS_OUT, windows=_WINDOWS, lags=_LAGS
        )

    if args.trades:
        df_trades = trades(df_y=df_y, days_out=_DAYS_OUT, tickers=_TICKERS)
        if args.sync:
            sync_positions(df_trades)


if __name__ == "__main__":
    main()
