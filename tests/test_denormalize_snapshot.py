"""
Snapshot regression cover for `denormalize`.

Seeded runs must stay bit-for-bit reproducible: callers rely on a fixed
`np.random` seed giving the same simulated path. The snapshots were generated
from the implementation before it was optimised, and every case also records
the next draw from `np.random` after the call, so any change in how many
random numbers `denormalize` consumes (or in what order) shows up as a failure.

Regenerate (only when an output change is intended) with:
    python -m tests.test_denormalize_snapshot
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from wavetrainer.model.model import PROBABILITY_COLUMN_PREFIX

from panelbeater.normalizer import denormalize

QUANTILES = 5
SEEDS = [0, 1, 42]
SNAPSHOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "snapshots", "denormalize.pkl"
)

# Quantile-format series. AAA has KDEs for most buckets (one bucket is None,
# so it falls back to the empirical-quantile uniform); EEE has KDEs whose
# accept window is so narrow that rejection sampling always fails; BBB has no
# KDEs at all.
QUANTILE_KDE = ["PX_AAA", "PX_EEE"]
QUANTILE_NO_KDE = ["PX_BBB"]
# Z-score-format series. "PX_AA" is a strict prefix of "PX_AAA", to cover
# prefix confusion in column lookup. DDD has no null column.
ZSCORE = ["PX_AA", "PX_CCC", "PX_DDD"]
# A series in y with no probability columns at all: it is skipped.
NO_PROBS = ["PX_FFF"]
SERIES = QUANTILE_KDE + QUANTILE_NO_KDE + ZSCORE + NO_PROBS
NULL_SERIES = ["PX_AAA", "PX_BBB", "PX_AA", "PX_CCC", "PX_EEE"]


def _build_case():  # pylint: disable=too-many-locals
    """Builds df (probabilities), y (prices) and KDEs covering every branch."""
    rng = np.random.default_rng(1234)
    dates = pd.date_range("2022-01-01", periods=420, freq="D")
    y = pd.DataFrame(
        {
            name: 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01 * (i + 1), 420))
            for i, name in enumerate(SERIES)
        },
        index=dates,
    )

    # Like the real caller, df carries the model's probabilities on its last
    # row. A second, older row checks that the last non-null value is used,
    # and two versions of each probability column check `sorted(...)[-1]`.
    df_dates = dates[[-3, -1]]
    data: dict[str, list[float]] = {}

    def add(prefix: str, old: float, new: float) -> None:
        data[f"{prefix}_{PROBABILITY_COLUMN_PREFIX}0"] = [new, old]
        data[f"{prefix}_{PROBABILITY_COLUMN_PREFIX}1"] = [old, new]

    for name in QUANTILE_KDE + QUANTILE_NO_KDE:
        weights = rng.uniform(0.1, 1.0, QUANTILES)
        for q in range(QUANTILES):
            add(f"{name}_q{q}", 0.5, float(weights[q]))
    for name in ZSCORE:
        stds = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.5]
        weights = rng.uniform(0.1, 1.0, len(stds))
        for std, w in zip(stds, weights):
            add(f"{name}_{std}", 0.5, float(w))
    for name in NULL_SERIES:
        add(f"{name}_null", 0.0, 0.3)

    df = pd.DataFrame(data, index=df_dates)

    kdes: dict = {}
    for name in QUANTILE_KDE:
        returns = y[name].pct_change().dropna().to_numpy()
        kdes[name] = {}
        for q in range(QUANTILES):
            if name == "PX_AAA" and q == 2:
                kdes[name][q] = None
                continue
            lo, hi = np.quantile(returns, [q / QUANTILES, (q + 1) / QUANTILES])
            if name == "PX_EEE":
                accept_lo, accept_hi = 10.0, 10.0  # never accepted
            else:
                accept_lo, accept_hi = lo, hi
            kdes[name][q] = {
                "kde": stats.gaussian_kde(returns),
                "min": float(accept_lo),
                "max": float(accept_hi),
                "fallback_min": float(lo),
                "fallback_max": float(hi),
            }
    return df, y, kdes


def _u_sample(case: str):
    if case == "omitted":
        return None
    if case == "full":
        return pd.Series({name: (i + 0.5) / len(SERIES) for i, name in enumerate(SERIES)})
    # A subset: the remaining series are sampled independently.
    return pd.Series({"PX_AAA": 0.9, "PX_CCC": 0.1, "PX_EEE": 0.5})


CASES = [
    (seed, u_case, with_kdes)
    for seed in SEEDS
    for u_case in ["omitted", "full", "subset"]
    for with_kdes in [True, False]
]


def _case_id(case) -> str:
    seed, u_case, with_kdes = case
    return f"seed{seed}-u_{u_case}-{'kdes' if with_kdes else 'nokdes'}"


def _run(case):
    """Runs a few chained calls, like consecutive simulated days."""
    seed, u_case, with_kdes = case
    df, y, kdes = _build_case()
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            y = denormalize(
                df.copy(),
                y,
                u_sample=_u_sample(u_case),
                quantiles=QUANTILES,
                precomputed_kdes=kdes if with_kdes else None,
            )
            # The real caller's df is dated on y's last day.
            df.index = pd.DatetimeIndex([y.index[-3], y.index[-1]])
    next_draw = np.random.random()
    return y, next_draw


def _load_snapshots() -> dict:
    with open(SNAPSHOT_PATH, "rb") as handle:
        return pickle.load(handle)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_denormalize_matches_snapshot(case):
    """Seeded output and random-number consumption match the snapshot exactly."""
    expected_df, expected_next_draw = _load_snapshots()[_case_id(case)]
    result, next_draw = _run(case)
    pd.testing.assert_frame_equal(result, expected_df, check_exact=True)
    assert result.index.freq == expected_df.index.freq
    # Same random-number consumption as the snapshot.
    assert next_draw == expected_next_draw


def test_snapshot_cases_exercise_null_and_skip_paths():
    """Guards against the snapshot quietly losing coverage."""
    snapshots = _load_snapshots()
    results = [snapshots[_case_id(c)][0] for c in CASES]
    last = pd.concat([r.iloc[-3:] for r in results])
    # Skipped series get NaN on new rows.
    assert last["PX_FFF"].isna().all()
    # Some null draws fired (exact 0% change) and some did not.
    flat = pd.concat([r.iloc[-4:].diff().iloc[1:] for r in results])
    assert (flat[NULL_SERIES] == 0.0).to_numpy().any()
    assert (flat[NULL_SERIES] != 0.0).to_numpy().any()


def _regenerate() -> None:
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    snapshots = {_case_id(case): _run(case) for case in CASES}
    with open(SNAPSHOT_PATH, "wb") as handle:
        pickle.dump(snapshots, handle)
    print(f"Wrote {len(snapshots)} snapshots to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    _regenerate()
