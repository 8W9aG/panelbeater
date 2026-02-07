"""Handles fitting models."""

# pylint: disable=too-many-locals,invalid-name
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
    horizons: list[int] | None = None,
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
    if horizons is not None:
        horizons = sorted(horizons)
        current_df_x = df_x.copy()
        for h in horizons:
            # 1. Fit Copula for THIS Horizon
            # We calculate specific h-day returns so the copula learns the correct
            # dependency structure (e.g., 30-day correlations vs 1-day correlations)
            returns_h = df_y.pct_change(periods=h).dropna()
            if isinstance(returns_h, pd.Series):
                returns_h = returns_h.to_frame()
            fit_vine_copula(returns_h, horizon=h)

            # 1. Normalize SPECIFICALLY for this horizon
            # This calculates the change over 'h' days (e.g., 32-day return)
            df_y_h_raw = normalize(df=df_y.copy(), horizon=h)

            # 2. Shift Backwards
            # The return calculated at day T represents (Price_T - Price_T-h).
            # We want to use features at T-h to predict this.
            # So we shift the target back by h.
            y_target = df_y_h_raw.shift(-h)

            # 3. Rename Target Columns
            # Appending _h{h} ensures we know this target is specifically for horizon h
            y_target.columns = [f"{col}_h{h}" for col in y_target.columns]

            # 4. Align Data
            valid_idx = y_target.dropna().index.intersection(current_df_x.index)

            X_train = current_df_x.loc[valid_idx]
            y_train = y_target.loc[valid_idx]

            # 5. Fit Model
            if fit_func is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    wavetrainer.fit(X_train, y=y_train)
            else:
                fit_func(X_train, y_train, wavetrainer)

            # 6. Chain Prediction
            current_df_x = wavetrainer.transform(current_df_x)
