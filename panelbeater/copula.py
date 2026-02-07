"""Handle joint distributions."""

# pylint: disable=too-many-locals,pointless-string-statement
import hashlib
import os
import pickle
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import pyvinecopulib as pv


def _vine_filename(df_returns: pd.DataFrame, horizon: int) -> str:
    """
    Generates a filename that is unique to:
    1. The specific assets in the dataframe (columns).
    2. The time horizon of the returns (e.g., h1 vs h30).
    """
    struct_str = hashlib.md5(
        "-".join(sorted(df_returns.columns.values.tolist())).encode("utf-8")
    ).hexdigest()
    # Added _h{horizon} to the filename
    return f"market_structure_{struct_str}_h{horizon}.pkl"


def load_vine_copula(df_returns: pd.DataFrame, horizon: int = 1) -> pv.Vinecop:
    """Loads a vine copula model for a specific horizon."""
    df_returns = df_returns.reindex(sorted(df_returns.columns), axis=1)
    filename = _vine_filename(df_returns=df_returns, horizon=horizon)

    with open(filename, "rb") as f:
        return pickle.load(f)


def fit_vine_copula(
    df_returns: pd.DataFrame, horizon: int = 1, ttl_days: int = 30
) -> pv.Vinecop:
    """
    Returns a fitted vine copula.
    Loads from disk if a valid (non-expired) model exists for this specific horizon;
    otherwise fits and saves.
    """
    df_returns = df_returns.reindex(sorted(df_returns.columns), axis=1)
    vine_file = _vine_filename(df_returns=df_returns, horizon=horizon)

    # 1. Check for valid cached model
    if os.path.exists(vine_file):
        file_age_seconds = time.time() - os.path.getmtime(vine_file)
        if file_age_seconds < (ttl_days * 24 * 60 * 60):
            print(f"Loading cached vine copula (horizon={horizon}) from {vine_file}")
            return load_vine_copula(df_returns=df_returns, horizon=horizon)

    # 2. If expired or missing, fit a new one
    print(
        f"Vine copula (horizon={horizon}) is missing or expired. Fitting new model..."
    )
    n = len(df_returns)
    # Manual PIT transform to Uniform [0, 1]
    u = df_returns.rank(method="average").values / (n + 1)

    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student],  # type: ignore
        tree_criterion="tau",
    )

    cop = pv.Vinecop.from_data(u, controls=controls)

    # 3. Save via Pickle
    with open(vine_file, "wb") as f:
        pickle.dump(cop, f, protocol=pickle.HIGHEST_PROTOCOL)

    return cop


def sample_joint_step(cop: pv.Vinecop) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Returns one joint sample vector for the panel."""
    simulated = np.array(cop.simulate(1))
    return cast(np.ndarray[Any, np.dtype[np.float64]], simulated[0])
