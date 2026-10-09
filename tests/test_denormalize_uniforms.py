"""
Tests that `denormalize` gives each series its OWN copula uniform.

Regression cover for the bug where uniforms were consumed positionally while
iterating a `set`, so the mapping was randomised per process by PYTHONHASHSEED
and the copula's dependence structure was permuted across columns.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
from wavetrainer.model.model import PROBABILITY_COLUMN_PREFIX

from panelbeater.normalizer import denormalize

SERIES = ["PX_AAA", "PX_BBB", "PX_CCC"]
QUANTILES = 4


class _FixedKDE:
    """A KDE stand-in that always returns one known value."""

    def __init__(self, value: float) -> None:
        self._value = value

    def resample(self, _n: int) -> np.ndarray:
        return np.array([[self._value]])


def _bucket_marker(series_idx: int, bucket: int) -> float:
    """A value that uniquely identifies (series, bucket) in the output."""
    return (series_idx + 1) * 0.01 + bucket * 0.001


def _build_case():
    """
    Builds df/y plus KDEs where the resulting price reveals exactly which
    bucket each series was assigned, with no randomness anywhere.
    """
    dates = pd.date_range("2022-01-01", periods=400, freq="D")
    y = pd.DataFrame(
        {name: np.linspace(100, 200, 400) for name in SERIES}, index=dates
    )

    df = pd.DataFrame(index=dates)
    for name in SERIES:
        # Uniform bucket probabilities: the chosen bucket is purely a function
        # of the uniform, which is what we want to observe.
        for q in range(QUANTILES):
            df[f"{name}_q{q}_{PROBABILITY_COLUMN_PREFIX}1"] = 1.0 / QUANTILES
        # Never force a null, so the sampled value survives to the output.
        df[f"{name}_null_{PROBABILITY_COLUMN_PREFIX}1"] = 0.0

    kdes = {
        name: {
            q: {
                "kde": _FixedKDE(_bucket_marker(i, q)),
                "min": -1.0,
                "max": 1.0,
                "fallback_min": 0.0,
                "fallback_max": 0.0,
            }
            for q in range(QUANTILES)
        }
        for i, name in enumerate(SERIES)
    }
    return df, y, kdes


def _buckets_from_result(result: pd.DataFrame, y: pd.DataFrame) -> dict:
    """Recovers which bucket each series landed in, from the final prices."""
    last_row = result.iloc[-1]
    out = {}
    for i, name in enumerate(SERIES):
        ret = last_row[name] / y[name].iloc[-1] - 1.0
        for q in range(QUANTILES):
            if np.isclose(ret, _bucket_marker(i, q), atol=1e-9):
                out[name] = q
                break
    return out


def test_each_series_uses_its_own_uniform():
    """Distinct uniforms must select the bucket matching each series' own value."""
    df, y, kdes = _build_case()

    # Chosen so each series lands in a different bucket: with 4 equal buckets,
    # cumulative edges are .25/.50/.75/1.0
    uniforms = pd.Series({"PX_AAA": 0.05, "PX_BBB": 0.40, "PX_CCC": 0.90})

    result = denormalize(
        df.copy(), y, u_sample=uniforms, quantiles=QUANTILES, precomputed_kdes=kdes
    )
    buckets = _buckets_from_result(result, y)

    assert buckets == {"PX_AAA": 0, "PX_BBB": 1, "PX_CCC": 3}


def test_permuting_uniforms_permutes_buckets():
    """Sanity: the mapping really is driven by the label, not by position."""
    df, y, kdes = _build_case()

    forward = denormalize(
        df.copy(),
        y,
        u_sample=pd.Series({"PX_AAA": 0.05, "PX_BBB": 0.40, "PX_CCC": 0.90}),
        quantiles=QUANTILES,
        precomputed_kdes=kdes,
    )
    reversed_ = denormalize(
        df.copy(),
        y,
        u_sample=pd.Series({"PX_AAA": 0.90, "PX_BBB": 0.40, "PX_CCC": 0.05}),
        quantiles=QUANTILES,
        precomputed_kdes=kdes,
    )

    assert _buckets_from_result(forward, y) == {"PX_AAA": 0, "PX_BBB": 1, "PX_CCC": 3}
    assert _buckets_from_result(reversed_, y) == {"PX_AAA": 3, "PX_BBB": 1, "PX_CCC": 0}


def test_array_with_u_columns_matches_labelled_series():
    """The ndarray + u_columns form must agree with the labelled form."""
    df, y, kdes = _build_case()
    values = [0.05, 0.40, 0.90]

    labelled = denormalize(
        df.copy(),
        y,
        u_sample=pd.Series(dict(zip(SERIES, values))),
        quantiles=QUANTILES,
        precomputed_kdes=kdes,
    )
    positional = denormalize(
        df.copy(),
        y,
        u_sample=np.array(values),
        u_columns=SERIES,
        quantiles=QUANTILES,
        precomputed_kdes=kdes,
    )

    assert _buckets_from_result(labelled, y) == _buckets_from_result(positional, y)


def test_unlabelled_array_is_rejected():
    """A bare array has no way to say which uniform belongs to which series."""
    df, y, kdes = _build_case()
    with pytest.raises(ValueError, match="unlabelled array"):
        denormalize(
            df.copy(),
            y,
            u_sample=np.array([0.05, 0.40, 0.90]),
            quantiles=QUANTILES,
            precomputed_kdes=kdes,
        )


def test_subset_does_not_borrow_another_series_uniform():
    """
    When the uniforms cover only some series, the rest must be sampled
    independently -- never handed a neighbour's uniform.
    """
    df, y, kdes = _build_case()

    # Only AAA and CCC are in the copula; BBB must not consume either uniform.
    uniforms = pd.Series({"PX_AAA": 0.05, "PX_CCC": 0.90})

    with pytest.warns(UserWarning, match="sampled independently"):
        result = denormalize(
            df.copy(), y, u_sample=uniforms, quantiles=QUANTILES, precomputed_kdes=kdes
        )

    buckets = _buckets_from_result(result, y)
    assert buckets["PX_AAA"] == 0
    assert buckets["PX_CCC"] == 3
    # BBB got *a* bucket, drawn independently, and crucially not AAA's or CCC's uniform.
    assert buckets["PX_BBB"] in range(QUANTILES)


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import numpy as np, pandas as pd
    sys.path.insert(0, {tests_dir!r})
    from test_denormalize_uniforms import _build_case, _buckets_from_result, SERIES, QUANTILES
    from panelbeater.normalizer import denormalize

    df, y, kdes = _build_case()
    u = pd.Series({{"PX_AAA": 0.05, "PX_BBB": 0.40, "PX_CCC": 0.90}})
    result = denormalize(df.copy(), y, u_sample=u, quantiles=QUANTILES, precomputed_kdes=kdes)
    print(json.dumps(_buckets_from_result(result, y)))
    """
)


@pytest.mark.parametrize("hashseed", ["0", "1", "12345"])
def test_mapping_is_stable_across_hash_seeds(hashseed, tmp_path):
    """
    The original bug only showed up across processes: `set` iteration order is
    salted by PYTHONHASHSEED. Same input must give the same mapping regardless.
    """
    import os

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    script = tmp_path / "run_case.py"
    script.write_text(_SUBPROCESS_SCRIPT.format(tests_dir=tests_dir))

    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=os.path.dirname(tests_dir),
    )
    import json

    assert json.loads(proc.stdout.strip()) == {
        "PX_AAA": 0,
        "PX_BBB": 1,
        "PX_CCC": 3,
    }
