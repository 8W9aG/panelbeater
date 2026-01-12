"""Handle simulations from the models."""
# pylint: disable=too-many-arguments,too-many-positional-arguments

from typing import Callable

import pandas as pd
import pyvinecopulib as pv
import tqdm
from joblib import Parallel, delayed

from .copula import load_vine_copula, sample_joint_step
from .features import features
from .normalizer import denormalize
from .wt import create_wt

SIMULATION_COLUMN = "simulation"
SIMULATION_FILENAME = "sims.parquet"


def run_single_simulation(
    sim_idx: int,
    df_y,
    days_out: int,
    windows: list[int],
    lags: list[int],
    vine_cop: pv.Vinecop,
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
        df_y = denormalize(df_next, y=df_y.copy(), u_sample=u_step)

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
) -> pd.DataFrame:
    """Simulate from trained models."""
    print(f"Starting {sims} simulations in parallel...")
    vine_cop = load_vine_copula(df_returns=df_y)
    print("Loaded vine copula")
    if sim_func is None:
        # n_jobs=-1 uses all available CPU cores
        all_sims = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(
                i, df_y.copy(), days_out, windows, lags, vine_cop
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
