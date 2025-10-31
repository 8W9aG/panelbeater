"""The CLI for finding mispriced options."""

import datetime

import requests_cache
import wavetrainer as wt

from .download import download
from .features import features
from .normalizer import normalize

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
    "EURUSD=X",
    "USDJPY=X",
    # Crypto
    "BTC-USD",
    "ETH-USD",
]
_MACROS = [
    "GDP",
    "UNRATE",
    "CPIAUCSL",
    "FEDFUNDS",
    "DGS10",
    "T10Y2Y",
    "M2SL",
    "VIXCLS",
    "DTWEXBGS",
    "INDPRO",
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


def main() -> None:
    """The main CLI function."""
    session = requests_cache.CachedSession("panelbeater-cache")
    wavetrainer = wt.create(
        "panelbeater-train",
        walkforward_timedelta=datetime.timedelta(days=7),
        validation_size=datetime.timedelta(days=365),
        test_size=datetime.timedelta(days=365),
        allowed_models={"catboost"},
        max_false_positive_reduction_steps=0,
    )
    df_y = download(tickers=_TICKERS, macros=_MACROS, session=session)
    df_x = features(df=df_y.copy(), windows=_WINDOWS, lags=_LAGS)
    df_y = normalize(df=df_y)
    wavetrainer.fit(df_x, y=df_y)


if __name__ == "__main__":
    main()
