"""Tests for the vine copula cache and the labelled-uniform API."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from panelbeater.copula import (VineFit, _vine_paths, fit_vine, load_vine,
                                sample_joint_step, uniforms_by_column)


@pytest.fixture(name="returns")
def returns_fixture():
    """A small, cheap-to-fit returns frame."""
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    base = rng.standard_normal(n)
    return pd.DataFrame(
        {
            "AAA": base + rng.standard_normal(n) * 0.3,
            "BBB": base * 0.8 + rng.standard_normal(n) * 0.4,
            "CCC": rng.standard_normal(n),
        },
        index=idx,
    )


def test_fit_writes_metadata_round_tripping_column_order(returns, tmp_path):
    """The sidecar records the exact column order the vine was fitted on."""
    fit = fit_vine(returns, cache_dir=str(tmp_path), namespace="ns")

    assert fit.columns == ["AAA", "BBB", "CCC"]

    _, meta_file = _vine_paths(returns, 1, str(tmp_path), "ns")
    with open(meta_file, "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    assert meta["columns"] == ["AAA", "BBB", "CCC"]
    assert meta["n_rows"] == len(returns)
    assert meta["namespace"] == "ns"
    assert meta["horizon"] == 1
    assert meta["content_hash"]
    assert meta["fitted_at"]

    # the loader hands the order back
    assert load_vine(returns, cache_dir=str(tmp_path), namespace="ns").columns == [
        "AAA",
        "BBB",
        "CCC",
    ]


def test_appended_rows_reuse_the_fit(returns, tmp_path):
    """New rows on the end do not invalidate a fit -- the fitted data still exists."""
    fit_vine(returns, cache_dir=str(tmp_path))

    extra = pd.DataFrame(
        {"AAA": [0.1, 0.2], "BBB": [0.3, 0.4], "CCC": [0.5, 0.6]},
        index=pd.date_range("2024-05-01", periods=2, freq="D"),
    )
    grown = pd.concat([returns, extra])

    loaded = load_vine(grown, cache_dir=str(tmp_path))
    assert loaded.columns == ["AAA", "BBB", "CCC"]


def test_changed_history_is_not_reused(returns, tmp_path):
    """A corrected historical row must not silently reuse the stale fit."""
    fit_vine(returns, cache_dir=str(tmp_path))

    tampered = returns.copy()
    tampered.iloc[0, 0] = tampered.iloc[0, 0] + 10.0

    with pytest.raises(ValueError, match="historical rows changed"):
        load_vine(tampered, cache_dir=str(tmp_path))


def test_dropped_rows_are_not_reused(returns, tmp_path):
    """Removing rows is not an append, so the fit no longer describes the data."""
    fit_vine(returns, cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="rows were removed"):
        load_vine(returns.iloc[:-5], cache_dir=str(tmp_path))


def test_namespaces_do_not_collide(returns, tmp_path):
    """Identical column names in two namespaces get two separate cache files."""
    a_pkl, a_meta = _vine_paths(returns, 1, str(tmp_path), "nfl")
    b_pkl, b_meta = _vine_paths(returns, 1, str(tmp_path), "aflw")

    assert a_pkl != b_pkl
    assert a_meta != b_meta

    fit_vine(returns, cache_dir=str(tmp_path), namespace="nfl")
    fit_vine(returns, cache_dir=str(tmp_path), namespace="aflw")

    assert os.path.exists(a_pkl)
    assert os.path.exists(b_pkl)


def test_changed_data_refits(returns, tmp_path):
    """fit_vine refits rather than returning a cached fit for different data."""
    first = fit_vine(returns, cache_dir=str(tmp_path))
    _, meta_file = _vine_paths(returns, 1, str(tmp_path), None)
    with open(meta_file, "r", encoding="utf-8") as handle:
        first_hash = json.load(handle)["content_hash"]

    changed = returns.copy()
    changed.iloc[5, 1] = changed.iloc[5, 1] + 3.0
    fit_vine(changed, cache_dir=str(tmp_path))

    with open(meta_file, "r", encoding="utf-8") as handle:
        second_hash = json.load(handle)["content_hash"]

    assert first_hash != second_hash
    assert isinstance(first, VineFit)


def test_sample_is_labelled_with_fitted_columns(returns, tmp_path):
    """VineFit.sample returns uniforms keyed by series name, in fit order."""
    fit = fit_vine(returns, cache_dir=str(tmp_path))
    sample = fit.sample()

    assert isinstance(sample, pd.Series)
    assert list(sample.index) == ["AAA", "BBB", "CCC"]
    assert ((sample >= 0.0) & (sample <= 1.0)).all()

    # unlabelled form still available for callers that want the raw vector
    raw = sample_joint_step(fit.copula)
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (3,)


def test_sample_rejects_mismatched_column_count(returns, tmp_path):
    """Naming the wrong number of columns is a bug, not something to guess at."""
    fit = fit_vine(returns, cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="one-to-one"):
        sample_joint_step(fit.copula, columns=["AAA", "BBB"])


def test_uniforms_by_column_forms():
    """Series, mapping and array+names all resolve; a bare array does not."""
    assert uniforms_by_column(None) is None

    expected = {"AAA": 0.1, "BBB": 0.2}
    assert uniforms_by_column(pd.Series([0.1, 0.2], index=["AAA", "BBB"])) == expected
    assert uniforms_by_column({"AAA": 0.1, "BBB": 0.2}) == expected
    assert uniforms_by_column(np.array([0.1, 0.2]), ["AAA", "BBB"]) == expected

    with pytest.raises(ValueError, match="unlabelled array"):
        uniforms_by_column(np.array([0.1, 0.2]))

    with pytest.raises(ValueError, match="one-to-one"):
        uniforms_by_column(np.array([0.1, 0.2]), ["AAA"])
