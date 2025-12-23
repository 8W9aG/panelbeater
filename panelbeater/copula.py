"""Handle joint distributions."""

import json
import os
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import pyvinecopulib as pv

VINE_FILE = "market_structure.json"


def fit_vine_copula(df_returns: pd.DataFrame, ttl_days: int = 30) -> pv.Vinecop:
    """
    Returns a fitted vine copula.
    Loads from disk if a valid (non-expired) model exists; otherwise fits and saves.
    """
    struct_str = "-".join(sorted(df_returns.columns.values.tolist()))
    vine_file = f"market_structure_{struct_str}.json"

    # 1. Check for valid cached model
    if os.path.exists(vine_file):
        file_age_seconds = time.time() - os.path.getmtime(vine_file)
        if file_age_seconds < (ttl_days * 24 * 60 * 60):
            print(f"Loading cached vine copula from {vine_file}")
            with open(vine_file, "r", encoding="utf8") as f:
                return pv.Vinecop.from_json(json.load(f))

    # 2. If expired or missing, fit a new one
    print("Vine copula is missing or expired. Fitting new model...")
    n = len(df_returns)
    # Manual PIT transform to Uniform [0, 1]
    u = df_returns.rank(method="average").values / (n + 1)

    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student],  # type: ignore
        tree_criterion="tau",
    )

    cop = pv.Vinecop.from_data(u, controls=controls)

    # 3. Save for future runs
    with open(vine_file, "w", encoding="utf8") as f:
        json.dump(cop.to_json(), f)

    return cop


def sample_joint_step(cop: pv.Vinecop) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Returns one joint sample vector for the panel."""
    simulated = np.array(cop.simulate(1))
    return cast(np.ndarray[Any, np.dtype[np.float64]], simulated[0])
