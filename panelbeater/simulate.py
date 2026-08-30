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
            if q == 0:
                q_data = returns[returns <= upper_bound]
                min_valid, max_valid = -1.0, upper_bound
            elif q == quantiles - 1:
                q_data = returns[returns >= lower_bound]
                min_valid, max_valid = lower_bound, float("inf")
            else:
                q_data = returns[(returns >= lower_bound) & (returns <= upper_bound)]
                min_valid, max_valid = lower_bound, upper_bound

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

    for _ in tqdm.tqdm(range(days_out), desc="Simulation Days"):
        # 1. Feature Engineering
        df_x = features(df=df_y.copy(), windows=windows, lags=lags)

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
