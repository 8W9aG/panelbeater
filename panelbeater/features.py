"""Generate features over a dataframe."""

# pylint: disable=too-many-locals,broad-exception-caught
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from feature_engine.datetime import DatetimeFeatures


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
        baseline_vol = daily_returns.rolling(max_window).std()

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
                sigma = roll.std()

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
                df[f"{col}_skew_{w}"] = roll.skew().fillna(0)
                df[f"{col}_kurt_{w}"] = roll.kurt().fillna(0)

                # --- 3. Volatility Regime Ratios ---
                current_vol = daily_returns.rolling(w).std()
                vol_ratio = (
                    current_vol.div(baseline_vol)
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )
                df[f"{col}_vol_regime_{w}"] = vol_ratio

                # --- 4. Bollinger Band Width (Volatility Squeeze) ---
                # 4 sigma covers ~95% of data (2 up, 2 down)
                bb_width = (4 * sigma) / mu.replace(0.0, np.nan)
                df[f"{col}_bbwidth_{w}"] = bb_width.fillna(0)

                # New normalized version (Min-Max bounded strictly between 0.0 and 1.0)
                # We calculate a temporary safe width so rolling min/max
                # avoids the overflow outliers
                safe_width = (4 * sigma) / (mu.abs() + 1e-8)
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

    return df


def _meta_ticker_feature(
    df: pd.DataFrame, lags: list[int], windows: list[int]
) -> pd.DataFrame:
    dfs = [df]
    for lag in lags:
        dfs.append(df.shift(lag).add_suffix(f"_lag{lag}"))
    for window in windows:
        dfs.append(df.rolling(window).mean().add_suffix(f"_rmean{window}"))  # type: ignore
        dfs.append(df.rolling(window).std().add_suffix(f"_rstd{window}"))  # type: ignore
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


def fetch_yfinance_earnings(ticker: str, limit: int = 100) -> pd.DatetimeIndex:
    """
    Safely fetch earnings dates. Returns an empty DatetimeIndex if
    the asset is an ETF (e.g., SPY) or has no earnings history.
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


def _compute_earnings_distances(
    target_idx: pd.DatetimeIndex, cal_idx: pd.DatetimeIndex, prefix: str
) -> pd.DataFrame:
    """
    Vectorized computation of days since last earnings and days until next earnings.
    """
    # --- Edge Case: Asset has no earnings (e.g., SPY) ---
    if cal_idx.empty:
        return pd.DataFrame(
            {f"{prefix}_days_since_earn": np.nan, f"{prefix}_days_until_earn": np.nan},
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
            f"{prefix}_days_since_earn": days_since,
            f"{prefix}_days_until_earn": days_until,
        },
        index=target_idx,
    )


def _apply_calendar_features(
    df: pd.DataFrame, original_cols: list[str], earnings_dict: dict
) -> pd.DataFrame:
    cal_features = []

    for col in original_cols:
        cal_idx = earnings_dict.get(col, pd.DatetimeIndex([]))

        # Calculate distances (will return NaNs for SPY/empty calendars)
        earn_df = _compute_earnings_distances(df.index, cal_idx, prefix=col)
        cal_features.append(earn_df)

    if cal_features:
        return pd.concat([df] + cal_features, axis=1)

    return df


def features(df: pd.DataFrame, windows: list[int], lags: list[int]) -> pd.DataFrame:
    """Generate features on a dataframe."""
    # 1. Deduplicate inputs to prevent column collisions
    windows = sorted(list(set(windows)))
    lags = sorted(list(set(lags)))

    original_cols = df.columns.values.tolist()

    # 2. Generate Cross-Sectional Features (Use ONLY original cols)
    df_xs = _cross_sectional_features(df[original_cols])  # pyright: ignore

    # 3. Generate Time-Series & Market Physics Features (modifies df in-place)
    df = _ticker_features(df=df, windows=windows)

    # 4. Generate Meta Features (Lags/Rolling of the expanded df)
    df = _meta_ticker_feature(df, lags=lags, windows=windows)

    # 5. Merge Cross-Sectional Features
    df = pd.concat([df, df_xs], axis=1)

    # 6. Datetime Features
    df = _dt_features(df=df)

    # 7. Dynamically generate the earnings dictionary with namespace filtering
    earnings_dict = {}
    for col in original_cols:
        if col.startswith("MACRO_"):
            # Macros do not have corporate earnings
            earnings_dict[col] = pd.DatetimeIndex([])
        else:
            # Strip 'PX_' prefix to query yfinance
            yf_ticker = col.replace("PX_", "", 1) if col.startswith("PX_") else col
            # Store it using the ORIGINAL column name so it maps back to the dataframe
            earnings_dict[col] = fetch_yfinance_earnings(yf_ticker)

    df = _apply_calendar_features(df, original_cols, earnings_dict)

    # 8. Final Clean up
    return df.drop(columns=original_cols).shift(1)
