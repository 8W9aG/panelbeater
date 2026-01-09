"""Handles fitting models."""

import warnings
from typing import Any, Callable

import pandas as pd

from .copula import fit_vine_copula
from .features import features
from .normalizer import normalize
from .wt import create_wt


def fit(
    df_y: pd.DataFrame,
    windows: list[int],
    lags: list[int],
    fit_func: Callable[[pd.DataFrame, pd.DataFrame, Any], None] | None = None,
) -> None:
    """Fit the models."""
    wavetrainer = create_wt()
    # Fit Vine Copula on historical returns
    # We use pct_change to capture the dependency of returns
    returns = df_y.pct_change().dropna()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    fit_vine_copula(returns)
    df_x = features(df=df_y.copy(), windows=windows, lags=lags)
    df_y_norm = normalize(df=df_y.copy())
    if fit_func is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            wavetrainer.fit(df_x, y=df_y_norm)
    else:
        fit_func(df_x, df_y_norm, wavetrainer)
