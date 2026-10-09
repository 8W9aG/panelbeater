"""Causal Hodrick-Prescott trend/cycle features."""

# pylint: disable=broad-exception-caught

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Hodrick-Prescott smoothing-strength "theories", spanning a responsive
# short-run trend up to a near-secular one. lambda = sigma_irregular^2 /
# sigma_trend^2 in the equivalent state-space model (see _hp_column_lambda).
_HP_LAMBDAS = [1e2, 1e5, 1e8, 1e11]


def _hp_column_lambda(col: str, s: pd.Series, lam: float) -> dict[str, pd.Series]:
    """
    Causal (one-sided) Hodrick-Prescott trend/cycle decomposition.

    The classic HP filter is a global penalized least-squares smoother: the
    trend at time t depends on the entire series, including points after t.
    King & Rebelo (1993) show that HP trend is exactly the Kalman SMOOTHER
    estimate of a "smooth trend" local-linear-trend state-space model with
    variance ratio lambda = sigma_irregular^2 / sigma_trend^2. Swapping in
    the Kalman FILTER instead gives the causal, one-sided analogue: the
    trend/slope at t depends only on data up to t, computed in a single
    O(n) forward pass rather than repeatedly refitting on trailing windows.

    lambda is a fixed smoothing-strength dial (as in standard HP filter
    usage), not something estimated from the data, so there is no
    hyperparameter-fitting lookahead either.
    """
    prefix = f"{col}_hp{lam:g}"
    trend_dev = pd.Series(np.nan, index=s.index)
    trend_slope = pd.Series(np.nan, index=s.index)
    cycle_ratio = pd.Series(np.nan, index=s.index)

    clean = s.ffill().bfill()
    if clean.isna().any() or clean.nunique() < 2:
        return {
            f"{prefix}_trend_dev": trend_dev,
            f"{prefix}_trend_slope": trend_slope,
            f"{prefix}_cycle_ratio": cycle_ratio,
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = UnobservedComponents(
                clean.to_numpy(),
                level="smooth trend",  # pyright: ignore[reportArgumentType]
            )
            res = mod.filter([1.0, 1.0 / lam])
    except Exception:  # noqa
        return {
            f"{prefix}_trend_dev": trend_dev,
            f"{prefix}_trend_slope": trend_slope,
            f"{prefix}_cycle_ratio": cycle_ratio,
        }

    level = clean.to_numpy()
    safe_level = np.where(np.abs(level) < 1e-8, np.nan, level)
    filtered_level = res.filtered_state[0]
    filtered_slope = res.filtered_state[1]

    # Diffuse-initialization burn-in: early states are unreliable.
    burn = max(res.loglikelihood_burn, 1)

    trend_dev.iloc[burn:] = (filtered_level / safe_level - 1)[burn:]
    trend_slope.iloc[burn:] = (filtered_slope / safe_level)[burn:]
    cycle_ratio.iloc[burn:] = ((level - filtered_level) / safe_level)[burn:]

    return {
        f"{prefix}_trend_dev": trend_dev,
        f"{prefix}_trend_slope": trend_slope,
        f"{prefix}_cycle_ratio": cycle_ratio,
    }


def _hp_features(df: pd.DataFrame, lambdas: list[float], n_jobs: int = 1) -> pd.DataFrame:
    """Causal Hodrick-Prescott trend/cycle features for each column."""
    tasks = [(col, df[col], lam) for col in df.columns for lam in lambdas]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_hp_column_lambda)(col, s, lam) for col, s, lam in tasks
    )

    out: dict[str, pd.Series] = {}
    for result in results:
        if result:
            out.update(result)
    return pd.DataFrame(out, index=df.index)
