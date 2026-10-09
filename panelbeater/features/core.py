"""Generate features over a dataframe."""

# pylint: disable=too-many-locals,broad-exception-caught
import warnings

import numpy as np
import pandas as pd
from feature_engine.datetime import DatetimeFeatures

from .calendar import (_apply_calendar_features, fetch_fred_release_dates,
                       fetch_yfinance_earnings)
from .hp import _HP_LAMBDAS, _hp_features
from .rolling import _stable_rolling_moment, _stable_rolling_std


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

                # EWMA -- recency-weighted counterpart to the SMA above.
                # Same "where is price relative to its own average" signal,
                # but weights decay geometrically so recent moves dominate.
                # adjust=False keeps it a pure recursive EMA, which is
                # inherently causal (only ever sees past + current data).
                ewma = s.ewm(span=w, adjust=False).mean()
                # Normalized as a deviation like the SMA, but guarded the way
                # MACD is rather than dividing by `s` directly: spread macros
                # (T10Y2Y) cross zero, which makes a bare `/ s` explode.
                safe_level = s.abs() + sigma.abs() + 1e-8
                df[f"{col}_ewma_{w}"] = ((ewma - s) / safe_level).fillna(0)

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
