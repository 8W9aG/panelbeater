"""The CLI for finding mispriced options."""

# pylint: disable=too-many-locals,use-dict-literal,invalid-name,line-too-long
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
    "SPY",  # SPDR S&P 500 ETF
    "QQQ",  # Invesco QQQ Trust
    "EEM",  # iShares MSCI Emerging Markets ETF
    "HYG",  # iShares iBoxx $ High Yield Corporate Bond ETF
    "TLT",  # iShares 20+ Year Treasury Bond ETF
    "SPYD",  # State Street SPDR Portfolio S&P 500 High Dividend ETF
    # Commodities
    "GC=F",  # Gold
    "CL=F",  # Crude Oil
    "SI=F",  # Silver
    # FX
    "EURUSD=X",  # EUR/USD
    "USDJPY=X",  # USD/JPY
    # Crypto
    "BTC-USD",  # Bitcoin USD Price
    "ETH-USD",  # Ethereum USD Price
]
_MACROS = [
    # FRED
    "GDP",  # Gross Domestic Product
    "UNRATE",  # Unemployment Rate
    "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
    "FEDFUNDS",  # Federal Funds Effective Rate
    "DGS10",  # Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis
    "T10Y2Y",  # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
    "M2SL",  # M2
    "VIXCLS",  # CBOE Volatility Index: VIX
    "DTWEXBGS",  # Nominal Broad U.S. Dollar Index
    "INDPRO",  # Industrial Production: Total Index
    # Indexes
    "^IRX",  # 13 WEEK TREASURY BILL
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
_DAYS_OUT = 32
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
