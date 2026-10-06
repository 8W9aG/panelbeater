"""Handle simulations from the models."""
# pylint: disable=too-many-arguments,too-many-positional-arguments

from typing import Any, Callable

import pandas as pd
import pyvinecopulib as pv
import tqdm
from joblib import Parallel, delayed
from scipy import stats

from .copula import load_vine_copula, sample_joint_step
from .features import features
from .normalizer import denormalize
from .wt import create_wt

SIMULATION_COLUMN = "simulation"
SIMULATION_FILENAME = "sims.parquet"


# run_single_simulation recomputes features() on the entire history every
# simulated day but only ever reads the last row (see below). These bound
# that recompute to trailing windows instead of the full ~10k-row history.
#
# The windows/lags-based features (_ticker_features, _meta_ticker_feature)
# only ever look back a bounded number of rows -- verified empirically
# exact (bit-identical to a full-history computation, once pandas'
# rolling std/skew/kurt numerical instability is worked around) at that
# bound. The worst case among the strictly-windowed features is a
# *triple*-nested one: bbwidth_w (rolling mean/std, w rows) ->
# bbwidthnorm_w (a second rolling min/max pass on top, w more rows) -> a
# meta rolling(W) pass on top of that (W more rows) needs 2*w + W - 2
# rows; for w == W == max(windows) that's 3*max(windows) - 2. Confirmed
# by direct measurement: 598 rows leaves a real (0.06) error at that
# feature, 599 drops it to 2e-9 (floating-point noise).
#
# MACD is different: its EMAs have unbounded (exponentially-decaying)
# memory like the HP filter's Kalman state, not a hard window, so it
# never becomes exactly bit-identical no matter the lookback -- but it
# converges much faster than the HP filter does. Measured: at the
# windows-only bound above (698), MACD_200 (the slowest configured EMA)
# still showed a real ~0.0008 error on a slow-converging series
# (T10Y2Y), shrinking to floating-point noise by ~1500-2000 rows (~10x
# the slow EMA's span) -- hence the 10*max(windows) term below.
#
# Computed from `windows`/`lags` (with a safety margin) rather than
# hardcoded, so it can't silently fall short if the windows config changes.
def _feature_lookback(windows: list[int], lags: list[int]) -> int:
    return (
        max(
            3 * max(windows) - 2,
            max(windows) + max(lags),
            10 * max(windows),
        )
        + 100
    )


# HP filter lookback, separate from the above: its Kalman state has no
# hard window, so truncating always costs some accuracy versus a
# full-history computation, unlike the windows/lags features above. That
# accuracy loss doesn't shrink to floating-point noise no matter how
# generous the lookback margin is -- it's a genuine tradeoff, measured
# directly rather than derived. For the largest (near-secular) lambda, a
# strongly trending series (e.g. SPY) is already accurate by ~2000 rows,
# but a mean-reverting series (e.g. T10Y2Y, FEDFUNDS) still shows real
# error (~0.03-0.07 on a typical trend_dev scale of a few tenths) at 2520
# rows, dropping to ~0.0002-0.001 by 5000. The HP filter itself is cheap
# even at full history, so this is set well above the windows/lags lookback.
_SIMULATION_HP_LOOKBACK = 5000


def precompute_kdes(df_y: pd.DataFrame, quantiles: int) -> dict:
    """Precomputes KDEs for all quantiles across all assets.

    Bucket boundaries are derived from the trailing 365-day return window,
    matching the rolling-window bucket definition normalize() uses to label
    training targets (see normalizer.py). Using the static full-history
    quantile instead would define "quantile q" differently at training time
    than at sampling time, causing samples to be drawn from the wrong bucket
    whenever the recent volatility regime differs from the full-history one.
    """
    kdes: dict[str, dict[int, dict[str, Any] | None]] = {}
    for col in df_y.columns:
        # Get the full historical percentage changes
        returns = df_y[col].pct_change().dropna()
        # Bucket boundaries reflect the current (most recent) regime
        recent_returns = returns.iloc[-365:]
        kdes[col] = {}

        for q in range(quantiles):
            lower_pctile = q / quantiles
            upper_pctile = (q + 1) / quantiles

            lower_bound = recent_returns.quantile(lower_pctile)
            upper_bound = recent_returns.quantile(upper_pctile)

            # Slice the data for this specific quantile
            min_valid: float
            max_valid: float
            if q == 0:
                q_data = returns[returns <= upper_bound]
                min_valid, max_valid = -1.0, float(upper_bound)
            elif q == quantiles - 1:
                q_data = returns[returns >= lower_bound]
                min_valid, max_valid = float(lower_bound), float("inf")
            else:
                q_data = returns[(returns >= lower_bound) & (returns <= upper_bound)]
                min_valid, max_valid = float(lower_bound), float(upper_bound)

            # Fit the KDE if there is enough variance
            if len(q_data) > 1 and q_data.std() > 0:
                kdes[col][q] = {
                    "kde": stats.gaussian_kde(q_data),
                    "min": min_valid,
                    "max": max_valid,
                    "fallback_min": lower_bound,
                    "fallback_max": upper_bound,
                }
            else:
                kdes[col][q] = None  # Fallback to uniform if data is weirdly flat

    return kdes


def run_single_simulation(
    sim_idx: int,
    df_y,
    days_out: int,
    windows: list[int],
    lags: list[int],
    vine_cop: pv.Vinecop,
    save_state_func: Callable[[int, pd.DataFrame, bool], Any] | None = None,
    # --- NEW PARAMETERS ---
    quantiles: int | None = None,
    precomputed_kdes: dict | None = None,
):
    """
    Encapsulates a single Monte Carlo path generation.
    """
    # Local copies for thread-safety (though joblib uses processes)
    df_y = df_y.copy()
    wavetrainer = create_wt()
    feature_lookback = _feature_lookback(windows, lags)

    for _ in tqdm.tqdm(range(days_out), desc="Simulation Days"):
        # 1. Feature Engineering
        df_x = features(
            df=df_y.copy(),
            windows=windows,
            lags=lags,
            lookback=feature_lookback,
            hp_lookback=_SIMULATION_HP_LOOKBACK,
        )

        # 2. Get Model Prediction (u_step sample from Copula)
        u_step = sample_joint_step(vine_cop)

        # 3. Transform and Denormalize to get next day prices
        df_next = wavetrainer.transform(df_x.iloc[[-1]], ignore_no_dates=True).drop(
            columns=df_x.columns.values.tolist()
        )

        # --- UPDATED: Pass quantiles and KDEs down to denormalize ---
        df_y = denormalize(
            df_next,
            y=df_y.copy(),
            u_sample=u_step,
            quantiles=quantiles,
            precomputed_kdes=precomputed_kdes,
        )

        if save_state_func is not None:
            save_state_func(sim_idx, df_y, False)

    # Mark the simulation index and return only the relevant tail (for memory efficiency)
    df_result = df_y.tail(days_out + 1).copy()
    df_result[SIMULATION_COLUMN] = sim_idx
    return df_result


def simulate(
    sims: int,
    df_y: pd.DataFrame,
    days_out: int,
    windows: list[int],
    lags: list[int],
    sim_func: Callable[
        [int, pd.DataFrame, int, list[int], list[int], pv.Vinecop], list[pd.DataFrame]
    ]
    | None = None,
    # --- NEW PARAMETERS ---
    quantiles: int | None = None,
) -> pd.DataFrame:
    """Simulate from trained models."""
    print(f"Starting {sims} simulations in parallel...")
    vine_cop = load_vine_copula(df_returns=df_y)
    print("Loaded vine copula")

    # --- NEW: Precompute the KDEs once before the parallel loop ---
    precomputed_kdes = None
    if quantiles is not None:
        print("Precomputing KDE distributions...")
        precomputed_kdes = precompute_kdes(df_y, quantiles)

    if sim_func is None:
        # n_jobs=-1 uses all available CPU cores
        all_sims = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(
                i,
                df_y.copy(),
                days_out,
                windows,
                lags,
                vine_cop,
                None,
                quantiles,
                precomputed_kdes,  # --- UPDATED: Pass to parallel worker ---
            )
            for i in tqdm.tqdm(range(sims), desc="Simulating")
        )
    else:
        all_sims = sim_func(sims, df_y.copy(), days_out, windows, lags, vine_cop)

    # Combine all simulations into one large DataFrame
    df_mc = pd.concat(all_sims)  # type: ignore
    df_mc.to_parquet(SIMULATION_FILENAME)
    return df_mc


def load_simulations() -> pd.DataFrame:
    """Load the rendered simulations."""
    return pd.read_parquet(SIMULATION_FILENAME)
