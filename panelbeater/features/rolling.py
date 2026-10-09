"""Numerically stable rolling statistics."""

# pylint: disable=too-many-locals

from typing import overload

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@overload
def _stable_rolling_std(data: pd.Series, window: int) -> pd.Series: ...


@overload
def _stable_rolling_std(data: pd.DataFrame, window: int) -> pd.DataFrame: ...


def _stable_rolling_std(
    data: pd.Series | pd.DataFrame, window: int
) -> pd.Series | pd.DataFrame:
    """
    Numerically stable rolling standard deviation (ddof=1), matching
    pandas' rolling(window).std() semantics (NaN until `window` valid
    observations are available), but recomputed from scratch per window
    instead of via pandas' incremental algorithm.

    pandas' rolling().std() maintains a running mean/sum-of-squares state
    that it updates incrementally as the window slides, for O(n) speed.
    Observed in practice: a single extreme historical value can leave that
    running state corrupted by accumulated floating-point error for
    thousands of subsequent rows, producing std values many orders of
    magnitude too large even though the actual window at that point is
    unremarkable (confirmed by recomputing the same window from scratch).
    This trades some speed (O(n * window) instead of O(n)) for that
    guarantee never happening.
    """
    is_series = isinstance(data, pd.Series)
    frame = data.to_frame() if isinstance(data, pd.Series) else data

    values = frame.to_numpy(dtype=float)
    n, m = values.shape[0], values.shape[1]
    result = np.full((n, m), np.nan)

    if n >= window:
        windows = sliding_window_view(values, window_shape=window, axis=0)
        result[window - 1 :, :] = np.std(windows, axis=-1, ddof=1)

    out = pd.DataFrame(result, index=frame.index, columns=frame.columns)
    return out.iloc[:, 0] if is_series else out


@overload
def _stable_rolling_moment(data: pd.Series, window: int, kind: str) -> pd.Series: ...


@overload
def _stable_rolling_moment(
    data: pd.DataFrame, window: int, kind: str
) -> pd.DataFrame: ...


def _stable_rolling_moment(
    data: pd.Series | pd.DataFrame, window: int, kind: str
) -> pd.Series | pd.DataFrame:
    """
    Numerically stable rolling skewness/kurtosis, matching pandas'
    rolling(window).skew()/.kurt() semantics and formula (the bias-corrected
    "adjusted Fisher-Pearson" skewness / excess kurtosis estimator, verified
    to match scipy.stats.skew(bias=False) / kurtosis(fisher=True, bias=False)
    to floating-point precision), but recomputed from scratch per window.

    pandas' rolling().skew()/.kurt() have the same class of issue as
    rolling().std() (see _stable_rolling_std): confirmed in practice that
    rolling(5).kurt() on a ~10k-row series gave 8.97 for a window whose
    correct kurtosis (independently verified via scipy) is exactly 5.0.
    kind: "skew" or "kurt".
    """
    is_series = isinstance(data, pd.Series)
    frame = data.to_frame() if isinstance(data, pd.Series) else data

    values = frame.to_numpy(dtype=float)
    n_rows, m = values.shape[0], values.shape[1]
    result = np.full((n_rows, m), np.nan)

    # Undefined (division by n-2 or (n-2)(n-3)) for windows this small --
    # pandas returns NaN here too rather than erroring.
    min_window = 3 if kind == "skew" else 4

    if n_rows >= window >= min_window:
        n = window
        win = sliding_window_view(values, window_shape=window, axis=0)
        mean = win.mean(axis=-1, keepdims=True)
        dev = win - mean
        m2 = np.mean(dev**2, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "skew":
                m3 = np.mean(dev**3, axis=-1)
                stat = (np.sqrt(n * (n - 1)) / (n - 2)) * (m3 / m2**1.5)
            else:
                m4 = np.mean(dev**4, axis=-1)
                g2_pop = m4 / m2**2 - 3
                stat = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2_pop + 6)
        result[window - 1 :, :] = stat

    out = pd.DataFrame(result, index=frame.index, columns=frame.columns)
    return out.iloc[:, 0] if is_series else out
