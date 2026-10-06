"""Generate features over a dataframe."""

# pylint: disable=too-many-locals,broad-exception-caught
import os
import warnings
from functools import lru_cache
from typing import overload

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from feature_engine.datetime import DatetimeFeatures
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.statespace.structural import UnobservedComponents

_FRED_API_ROOT = "https://api.stlouisfed.org/fred"

# Hodrick-Prescott smoothing-strength "theories", spanning a responsive
# short-run trend up to a near-secular one. lambda = sigma_irregular^2 /
# sigma_trend^2 in the equivalent state-space model (see _hp_column_lambda).
_HP_LAMBDAS = [1e2, 1e5, 1e8, 1e11]


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


def _macd_features(s: pd.Series, sigma: pd.Series, w: int) -> dict[str, pd.Series]:
    """MACD line, signal and histogram for one series at window `w`.

    Treat w as the "slow" EMA period; fast/signal periods approximate the
    classic 12/26/9 ratios (fast ~= 0.46*slow, signal ~= 0.35*slow) so each
    configured window gets its own fast/slow/signal MACD, consistent with how
    every other feature here is parameterized by `w`. EMAs are inherently
    causal (recursive, only ever see past + current data).

    Returns the three series keyed by column suffix.
    """
    fast_span = max(2, w // 2)
    signal_span = max(2, w // 3)
    ema_fast = s.ewm(span=fast_span, adjust=False).mean()
    ema_slow = s.ewm(span=w, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Normalize by price level (guard near-zero denominators -- e.g. spread
    # macros like T10Y2Y crossing zero). A bare `s.abs() + 1e-8` epsilon isn't
    # enough here -- confirmed the same failure bbwidth originally had: T10Y2Y
    # flattened to near-zero during its 2022 inversion and blew this up to
    # 1.7e7. `sigma` (the series' own rolling std) is automatically
    # scale-matched -- negligible next to `s` for a normal price series, but a
    # meaningful floor when `s` itself approaches zero.
    safe_s = s.abs() + sigma.abs() + 1e-8
    return {
        "macd": (macd_line / safe_s).fillna(0),
        "macdsignal": (signal_line / safe_s).fillna(0),
        "macdhist": (macd_hist / safe_s).fillna(0),
    }


def _ticker_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    cols = df.columns.values.tolist()

    # Establish the maximum window to act as our long-term baseline for volatility
    max_window = max(windows) if windows else 1

    # --- OPTIMIZATION: Calculate all returns and the market index once ---
    all_daily_returns = df.pct_change(fill_method=None)
    market_index = all_daily_returns.mean(axis=1)

    for col in cols:
        s = df[col]

        # Slice the pre-calculated returns for this ticker
        daily_returns = all_daily_returns[col]

        # Pre-calculate baseline long-term volatility (standard deviation of returns)
        baseline_vol = _stable_rolling_std(daily_returns, max_window)

        for w in windows:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                roll = s.rolling(w)

                # SMA
                sma = roll.mean()
                df[f"{col}_sma_{w}"] = sma / s - 1

                # PCT
                df[f"{col}_pctchg_{w}"] = s.pct_change(w, fill_method=None)

                # Z-Score
                mu = roll.mean()
                sigma = _stable_rolling_std(s, w)

                # Protect against zero variance
                safe_sigma = sigma.replace(0.0, np.nan)
                df[f"{col}_z_{w}"] = ((s - mu) / safe_sigma).fillna(0.0)

                # --- 1. Kaufman Efficiency Ratio (Fractal Efficiency) ---
                change = s.diff(w).abs()
                volatility = s.diff(1).abs().rolling(w).sum()
                # Protect volatility division BEFORE the replace/fillna
                er = change.div(volatility.replace(0.0, np.nan)).fillna(0)
                df[f"{col}_er_{w}"] = er

                # --- 2. Rolling Skewness & Kurtosis ---
                df[f"{col}_skew_{w}"] = _stable_rolling_moment(s, w, "skew").fillna(0)
                df[f"{col}_kurt_{w}"] = _stable_rolling_moment(s, w, "kurt").fillna(0)

                # --- 3. Volatility Regime Ratios ---
                current_vol = _stable_rolling_std(daily_returns, w)
                vol_ratio = (
                    current_vol.div(baseline_vol)
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )
                df[f"{col}_vol_regime_{w}"] = vol_ratio

                # --- 4. Bollinger Band Width (Volatility Squeeze) ---
                # 4 sigma covers ~95% of data (2 up, 2 down)
                # Guard against near-zero mu (e.g. T10Y2Y crossing zero during
                # yield curve inversions), which otherwise blows this up to
                # extreme values -- same guard bbwidthnorm below already used.
                safe_width = (4 * sigma) / (mu.abs() + 1e-8)
                df[f"{col}_bbwidth_{w}"] = safe_width.fillna(0)

                # New normalized version (Min-Max bounded strictly between 0.0 and 1.0)
                roll_min = safe_width.rolling(w).min()
                roll_max = safe_width.rolling(w).max()
                bb_width_norm = (safe_width - roll_min) / (roll_max - roll_min + 1e-8)
                df[f"{col}_bbwidthnorm_{w}"] = bb_width_norm.fillna(0)

                # --- 5. Rolling Autocorrelation (Serial Correlation) ---
                autocorr = daily_returns.rolling(w).corr(daily_returns.shift(1))
                df[f"{col}_autocorr_{w}"] = autocorr.fillna(0)

                # --- 6. RSI (Relative Strength Index) - Vectorized ---
                diff = s.diff(1)
                gain = diff.clip(lower=0).rolling(w).mean()
                loss = -1 * diff.clip(upper=0).rolling(w).mean()

                # Safe division for RS
                rs = gain / loss.replace(0, np.nan)  # type: ignore
                rsi = 1 - (1 / (1 + rs))
                # If loss is 0 (no down days), RSI is 1.0 (100 in standard RSI terms)
                df[f"{col}_rsi_{w}"] = rsi.fillna(1.0)

                # --- 7. Market Physics (Rolling Beta) ---
                # "Is this asset decoupled from the market?"
                beta_proxy = daily_returns.rolling(w).corr(market_index)  # type: ignore
                df[f"{col}_beta_{w}"] = beta_proxy.fillna(0)

                # --- 8. MACD (Moving Average Convergence Divergence) ---
                for suffix, macd_series in _macd_features(s, sigma, w).items():
                    df[f"{col}_{suffix}_{w}"] = macd_series

    return df


def _meta_ticker_feature(
    df: pd.DataFrame, lags: list[int], windows: list[int]
) -> pd.DataFrame:
    dfs = [df]
    for lag in lags:
        dfs.append(df.shift(lag).add_suffix(f"_lag{lag}"))
    for window in windows:
        dfs.append(df.rolling(window).mean().add_suffix(f"_rmean{window}"))  # type: ignore
        dfs.append(_stable_rolling_std(df, window).add_suffix(f"_rstd{window}"))
    return pd.concat(dfs, axis=1).replace([np.inf, -np.inf], np.nan)


def _dt_features(df: pd.DataFrame) -> pd.DataFrame:
    dtf = DatetimeFeatures(features_to_extract="all", variables="index")
    return dtf.fit_transform(df)


def _cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates relative strength features across the panel at each timestamp.
    """
    daily_returns = df.pct_change(fill_method=None)

    # 1. Cross-Sectional Rank (0.0 to 1.0)
    cs_rank = daily_returns.rank(axis=1, pct=True).add_suffix("_xs_rank")

    # 2. Cross-Sectional Z-Score (Distance from Market Mean)
    mean = daily_returns.mean(axis=1)
    std = daily_returns.std(axis=1)

    # --- FIX: Prevent ZeroDivisionError on stagnant days ---
    # Replace 0.0 with NaN so division yields NaN instead of an exception,
    # then fill the NaN z-scores back to 0.0
    safe_std = std.replace(0.0, np.nan)
    cs_z = daily_returns.sub(mean, axis=0).div(safe_std, axis=0).fillna(0.0)
    cs_z = cs_z.add_suffix("_xs_z")

    return pd.concat([cs_rank, cs_z], axis=1)


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


@lru_cache(maxsize=None)
def fetch_yfinance_earnings(ticker: str, limit: int = 100) -> pd.DatetimeIndex:
    """
    Safely fetch earnings dates. Returns an empty DatetimeIndex if
    the asset is an ETF (e.g., SPY) or has no earnings history.

    Cached per-process: this depends only on the ticker, not on the price
    data or the date range being featurized, but features() is called once
    per simulated day inside run_single_simulation's loop (see simulate.py)
    -- without caching, that refetches the same calendar over the network
    on every single day-step of every simulation.
    """
    try:
        t = yf.Ticker(ticker)

        # Suppress yfinance stdout/warnings for cleaner bulk processing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dates_df = t.get_earnings_dates(limit=limit)

        # Handle SPY, indices, or missing data
        if dates_df is None or dates_df.empty:
            return pd.DatetimeIndex([])

        # Strip timezone and normalize to midnight for exact date alignment
        return (
            pd.to_datetime(dates_df.index).tz_localize(None).normalize().sort_values()
        )

    except Exception:  # noqa
        # Catch internal yfinance parsing errors on unsupported assets
        return pd.DatetimeIndex([])


@lru_cache(maxsize=None)
def fetch_fred_release_dates(series_code: str, limit: int = 100) -> pd.DatetimeIndex:
    """
    Safely fetch a FRED series' release calendar (past and publicly
    scheduled future release/announcement dates, e.g. CPI or FOMC dates).
    Returns an empty DatetimeIndex if the series has no associated FRED
    release (e.g. Yahoo-sourced index macros like ^IRX) or the lookup fails.

    Cached per-process: see fetch_yfinance_earnings for why this matters.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key or series_code.startswith("^"):
        return pd.DatetimeIndex([])

    try:
        release_resp = requests.get(
            f"{_FRED_API_ROOT}/series/release",
            params={
                "series_id": series_code,
                "api_key": api_key,
                "file_type": "json",
            },
            timeout=10,
        )
        release_resp.raise_for_status()
        releases = release_resp.json().get("releases", [])
        if not releases:
            return pd.DatetimeIndex([])
        release_id = releases[0]["id"]

        dates_resp = requests.get(
            f"{_FRED_API_ROOT}/release/dates",
            params={
                "release_id": release_id,
                "api_key": api_key,
                "file_type": "json",
                "include_release_dates_with_no_data": "true",
                "limit": limit,
                "sort_order": "desc",
            },
            timeout=10,
        )
        dates_resp.raise_for_status()
        release_dates = dates_resp.json().get("release_dates", [])
        if not release_dates:
            return pd.DatetimeIndex([])

        return (
            pd.to_datetime([d["date"] for d in release_dates]).normalize().sort_values()
        )

    except Exception:  # noqa
        # Catch network/parsing errors on unsupported or misconfigured series
        return pd.DatetimeIndex([])


def _compute_calendar_distances(
    target_idx: pd.DatetimeIndex, cal_idx: pd.DatetimeIndex, prefix: str, label: str
) -> pd.DataFrame:
    """
    Vectorized computation of days since last and days until next calendar event.
    """
    # --- Edge Case: Asset has no calendar events (e.g., SPY earnings, ^IRX releases) ---
    if cal_idx.empty:
        return pd.DataFrame(
            {
                f"{prefix}_days_since_{label}": np.nan,
                f"{prefix}_days_until_{label}": np.nan,
            },
            index=target_idx,
        )

    # Prepare dataframes for merge_asof (must be sorted)
    df_target = pd.DataFrame(index=target_idx.normalize()).sort_index()
    df_cal = pd.DataFrame({"cal_date": cal_idx}, index=cal_idx).sort_index()

    # Backward look: Most recent past date
    past = pd.merge_asof(
        df_target, df_cal, left_index=True, right_index=True, direction="backward"
    )
    days_since = (df_target.index - past["cal_date"]).dt.days

    # Forward look: Next upcoming date
    future = pd.merge_asof(
        df_target, df_cal, left_index=True, right_index=True, direction="forward"
    )
    days_until = (future["cal_date"] - df_target.index).dt.days

    return pd.DataFrame(
        {
            f"{prefix}_days_since_{label}": days_since,
            f"{prefix}_days_until_{label}": days_until,
        },
        index=target_idx,
    )


def _apply_calendar_features(
    df: pd.DataFrame, original_cols: list[str], cal_dict: dict, label: str
) -> pd.DataFrame:
    cal_features = []

    for col in original_cols:
        cal_idx = cal_dict.get(col, pd.DatetimeIndex([]))

        # Calculate distances (will return NaNs for empty calendars)
        cal_df = _compute_calendar_distances(
            pd.DatetimeIndex(df.index), cal_idx, prefix=col, label=label
        )
        cal_features.append(cal_df)

    if cal_features:
        return pd.concat([df] + cal_features, axis=1)

    return df


def features(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    df: pd.DataFrame,
    windows: list[int],
    lags: list[int],
    n_jobs: int = 1,
    lookback: int | None = None,
    hp_lookback: int | None = None,
) -> pd.DataFrame:
    """Generate features on a dataframe.

    lookback: if set, only the trailing `lookback` rows are used for the
    windows/lags-based features (_ticker_features, _meta_ticker_feature).
    These only ever look back a bounded number of rows in the worst case
    (2*max(windows)-1, for a meta rolling window applied on top of an
    already max(windows)-rolled ticker feature) -- once lookback exceeds
    that, results are exactly identical (verified empirically, not just
    assumed) to a full-history computation.

    hp_lookback: separately controls how much history the HP filter
    (_hp_features) sees. Unlike the windows/lags features, the Kalman
    filter's state is a weighted combination of ALL prior observations
    with no hard window, so truncating always introduces some divergence
    from a full-history computation -- how much depends on how slowly the
    filter's gain converges for a given lambda and series. Measured:
    near-instant for a strongly trending series (e.g. SPY), but the
    largest lambda on a mean-reverting series (e.g. T10Y2Y, FEDFUNDS)
    needs several thousand rows before the error becomes small. Defaults
    to `lookback` if not given; pass a larger value (or None for full
    history) when accuracy on such series matters more than the modest
    extra compute -- the HP filter is cheap even at full history, so this
    should usually be set well above `lookback`, not left to default to it.

    Row-count-independent steps (cross-sectional, calendar, datetime
    features) are unaffected by either. Leave both as None (full history,
    exact throughout) for training/backtesting.
    """
    # 1. Deduplicate inputs to prevent column collisions
    windows = sorted(list(set(windows)))
    lags = sorted(list(set(lags)))

    hp_lookback = lookback if hp_lookback is None else hp_lookback
    hp_df = df if hp_lookback is None else df.tail(hp_lookback)

    if lookback is not None:
        df = df.tail(lookback)

    original_cols = df.columns.values.tolist()

    # 2. Generate Cross-Sectional Features (Use ONLY original cols)
    df_xs = _cross_sectional_features(df[original_cols])  # pyright: ignore

    # 2b. Generate causal HP filter trend/cycle features (Use ONLY original
    # cols, and its own possibly-larger lookback -- see docstring)
    df_hp = _hp_features(  # pyright: ignore
        hp_df[original_cols], lambdas=_HP_LAMBDAS, n_jobs=n_jobs
    )
    df_hp = df_hp.reindex(df.index)

    # 3. Generate Time-Series & Market Physics Features (modifies df in-place)
    df = _ticker_features(df=df, windows=windows)

    # 4. Generate Meta Features (Lags/Rolling of the expanded df)
    df = _meta_ticker_feature(df, lags=lags, windows=windows)

    # 5. Merge Cross-Sectional & HP Filter Features
    df = pd.concat([df, df_xs, df_hp], axis=1)

    # 6. Datetime Features
    df = _dt_features(df=df)

    # 7. Dynamically generate the earnings and macro-release dictionaries
    # with namespace filtering
    earnings_dict = {}
    macro_dict = {}
    for col in original_cols:
        if col.startswith("MACRO_"):
            # Macros do not have corporate earnings
            earnings_dict[col] = pd.DatetimeIndex([])
            # Strip 'MACRO_' prefix to query the FRED release calendar
            fred_code = col.replace("MACRO_", "", 1)
            macro_dict[col] = fetch_fred_release_dates(fred_code)
        else:
            # Strip 'PX_' prefix to query yfinance
            yf_ticker = col.replace("PX_", "", 1) if col.startswith("PX_") else col
            # Store it using the ORIGINAL column name so it maps back to the dataframe
            earnings_dict[col] = fetch_yfinance_earnings(yf_ticker)
            macro_dict[col] = pd.DatetimeIndex([])

    df = _apply_calendar_features(df, original_cols, earnings_dict, label="earn")
    df = _apply_calendar_features(df, original_cols, macro_dict, label="macro")

    # 8. Final Clean up
    return df.drop(columns=original_cols).shift(1)
