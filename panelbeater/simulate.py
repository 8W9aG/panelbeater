"""Handle simulations from the models."""
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals

from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed
from scipy import stats

from .copula import VineFit, load_vine, vine_returns
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


def run_simulation_batch(
    sim_indices: list[int],
    df_y: pd.DataFrame,
    days_out: int,
    windows: list[int],
    lags: list[int],
    vine_fit: VineFit,
    save_state_func: Callable[[int, pd.DataFrame, bool], Any] | None = None,
    quantiles: int | None = None,
    precomputed_kdes: dict | None = None,
    n_jobs: int = 1,
) -> list[pd.DataFrame]:
    """
    Generates one Monte Carlo path per entry of `sim_indices`, advanced in lockstep.

    Every path visits the same dates in the same order: denormalize always
    appends index[-1] + 1 day, and wavetrainer picks each row's walk-forward
    fold from its date alone. So each day the paths' feature rows can be
    stacked into a single transform() call, which returns exactly what
    per-row calls do. That call's cost is almost all fixed per trained column
    (pandas overhead and each column's reducer/selector/model pipeline), not
    per row: measured on ~480 columns, one row took ~102s and six stacked rows
    ~109s. So one call per day for the whole batch pays it once instead of
    once per path. The batch also shares one Trainer and its model cache.

    The paths stay independent: each keeps its own df_y, copula draw and
    denormalize sample. features() is still per path, ~30s each per day.

    `n_jobs` is transform()'s own column-level parallelism. It defaults to 1
    because simulate() already runs batches in parallel; raise it only when
    calling this directly with fewer batches than cores.
    """
    # Every path starts from the same history and then grows its own.
    path_ys = [df_y.copy() for _ in sim_indices]
    # One Trainer for the whole batch, so its model cache is filled once and
    # reused every day by every path.
    wavetrainer = create_wt()
    feature_lookback = _feature_lookback(windows, lags)

    for _ in tqdm.tqdm(range(days_out), desc="Simulation Days"):
        # 1. Feature Engineering, one last row per path. Only the last row is
        # needed (it predicts the next day); [[-1]] keeps it a one-row frame
        # for the concat below. features() is written to modify the frame it is
        # given (_ticker_features works in place), so it gets a copy rather
        # than the path's own history.
        x_rows = [
            features(
                df=path_y.copy(),
                windows=windows,
                lags=lags,
                lookback=feature_lookback,
                hp_lookback=_SIMULATION_HP_LOOKBACK,
            ).iloc[[-1]]
            for path_y in path_ys
        ]

        # 2. One stacked model prediction for the whole batch. All rows share
        # one date (the duplicate index is fine), so every row uses the same
        # walk-forward fold and the result matches per-row calls exactly.
        # concat keeps path order, so row i of df_next belongs to path i.
        x_batch = pd.concat(x_rows)
        # n_jobs defaults to 1: simulate() already runs batches in parallel,
        # and transform()'s own pool inside each worker would oversubscribe
        # the machine (up to cores^2 processes).
        df_next = wavetrainer.transform(
            x_batch, ignore_no_dates=True, n_jobs=n_jobs
        ).drop(columns=x_batch.columns.values.tolist())

        # 3. Sample from the copula and denormalize to get each path's next day.
        # Each path draws once per day, in path order. With one path this is
        # the same draw sequence as the old one-path loop, so its output is
        # unchanged; with several, the paths' draws interleave.
        for i, sim_idx in enumerate(sim_indices):
            u_step = vine_fit.sample()
            # denormalize only reads y, so the path's frame needs no copy.
            path_ys[i] = denormalize(
                df_next.iloc[[i]],
                y=path_ys[i],
                u_sample=u_step,
                quantiles=quantiles,
                precomputed_kdes=precomputed_kdes,
            )

            if save_state_func is not None:
                save_state_func(sim_idx, path_ys[i], False)

    # Mark the simulation index and return only the relevant tail (for memory efficiency)
    results = []
    for sim_idx, path_y in zip(sim_indices, path_ys):
        df_result = path_y.tail(days_out + 1).copy()
        df_result[SIMULATION_COLUMN] = sim_idx
        results.append(df_result)
    return results


def run_single_simulation(
    sim_idx: int,
    df_y,
    days_out: int,
    windows: list[int],
    lags: list[int],
    vine_fit: VineFit,
    save_state_func: Callable[[int, pd.DataFrame, bool], Any] | None = None,
    quantiles: int | None = None,
    precomputed_kdes: dict | None = None,
    n_jobs: int = 1,
):
    """
    Encapsulates a single Monte Carlo path generation.

    simulate() no longer calls this, it runs batches. It is kept, as a batch of
    one, for external callers (panelbeater-modal runs one path per container).
    Such a caller can pass n_jobs=-1 to spread transform() over the trained
    columns instead, since nothing else is using the machine's cores.
    """
    return run_simulation_batch(
        [sim_idx],
        df_y,
        days_out,
        windows,
        lags,
        vine_fit,
        save_state_func=save_state_func,
        quantiles=quantiles,
        precomputed_kdes=precomputed_kdes,
        n_jobs=n_jobs,
    )[0]


def simulate(
    sims: int,
    df_y: pd.DataFrame,
    days_out: int,
    windows: list[int],
    lags: list[int],
    sim_func: Callable[
        [int, pd.DataFrame, int, list[int], list[int], VineFit], list[pd.DataFrame]
    ]
    | None = None,
    # --- NEW PARAMETERS ---
    quantiles: int | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Simulate from trained models.

    `n_jobs` is how many batches run at once (joblib's convention, -1 for one
    per core). Each worker holds every trained column's models (~5GB) and
    features() briefly needs ~13GB more, so on a machine with less memory than
    cores x that, pass fewer.
    """
    print(f"Starting {sims} simulations in parallel...")
    # Load with the same returns frame fit() fitted on: the cache is validated
    # against its contents, so loading with prices would be rejected.
    vine_fit = load_vine(df_returns=vine_returns(df_y))
    print("Loaded vine copula")

    # --- NEW: Precompute the KDEs once before the parallel loop ---
    precomputed_kdes = None
    if quantiles is not None:
        print("Precomputing KDE distributions...")
        precomputed_kdes = precompute_kdes(df_y, quantiles)

    if sim_func is None:
        # One lockstep batch per worker: see run_simulation_batch for why
        # batching paths beats one path per task. Fewer, larger batches are
        # cheaper (transform() costs about the same for one row as for many),
        # so there are exactly as many batches as workers. min() avoids empty
        # batches when there are fewer paths than workers.
        batches = np.array_split(
            np.arange(sims), min(sims, joblib.effective_n_jobs(n_jobs))
        )
        # n_jobs is deliberately not passed into the batches: each keeps
        # transform() at n_jobs=1 (see run_simulation_batch).
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(run_simulation_batch)(
                batch.tolist(),
                df_y.copy(),
                days_out,
                windows,
                lags,
                vine_fit,
                None,
                quantiles,
                precomputed_kdes,
            )
            for batch in tqdm.tqdm(batches, desc="Simulating")
        )
        # Batches are contiguous ranges returned in order, so flattening keeps
        # paths in simulation-index order, as one task per path did.
        all_sims = [df for batch in batch_results for df in batch]  # type: ignore
    else:
        all_sims = sim_func(sims, df_y.copy(), days_out, windows, lags, vine_fit)

    # Combine all simulations into one large DataFrame
    df_mc = pd.concat(all_sims)  # type: ignore
    df_mc.to_parquet(SIMULATION_FILENAME)
    return df_mc


def load_simulations() -> pd.DataFrame:
    """Load the rendered simulations."""
    return pd.read_parquet(SIMULATION_FILENAME)
