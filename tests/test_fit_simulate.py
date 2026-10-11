"""fit() and simulate() must agree on the data the vine copula is keyed on."""

import importlib

import numpy as np
import pandas as pd
import pytest

# The package re-exports functions with the same names as these modules.
fit_module = importlib.import_module("panelbeater.fit")
simulate_module = importlib.import_module("panelbeater.simulate")


@pytest.fixture(name="prices")
def prices_fixture():
    """A small panel of positive, correlated price series."""
    rng = np.random.default_rng(0)
    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    base = rng.standard_normal(n) * 0.01
    moves = pd.DataFrame(
        {
            "PX_AAA": base + rng.standard_normal(n) * 0.005,
            "PX_BBB": base * 0.8 + rng.standard_normal(n) * 0.006,
            "PX_CCC": rng.standard_normal(n) * 0.01,
        },
        index=idx,
    )
    return 100.0 * (1.0 + moves).cumprod()


def test_simulate_loads_the_vine_fit_fitted(prices, tmp_path, monkeypatch):
    """simulate() finds and validates the vine that fit() just wrote.

    Training, features and the simulated paths are stubbed out: this is about
    the vine cache, which fit() and simulate() both key on the returns frame.
    """
    monkeypatch.chdir(tmp_path)  # the vine cache and sims.parquet go in the cwd
    monkeypatch.setattr(fit_module, "create_wt", lambda n_jobs=1: None)
    monkeypatch.setattr(fit_module, "features", lambda df, windows, lags: df)
    monkeypatch.setattr(fit_module, "normalize", lambda df, **kwargs: df)

    fit_module.fit(df_y=prices, windows=[5], lags=[1], fit_func=lambda *args: None)

    seen = {}

    def sim_func(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        sims, df_y, days_out, windows, lags, vine_fit
    ):
        del sims, days_out, windows, lags
        seen["columns"] = vine_fit.columns
        return [df_y.tail(1).assign(simulation=0)]

    simulate_module.simulate(
        sims=1, df_y=prices, days_out=1, windows=[5], lags=[1], sim_func=sim_func
    )

    assert seen["columns"] == ["PX_AAA", "PX_BBB", "PX_CCC"]


def test_simulate_still_loads_after_new_rows(prices, tmp_path, monkeypatch):
    """A day of new prices after the fit does not invalidate the vine."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fit_module, "create_wt", lambda n_jobs=1: None)
    monkeypatch.setattr(fit_module, "features", lambda df, windows, lags: df)
    monkeypatch.setattr(fit_module, "normalize", lambda df, **kwargs: df)

    fit_module.fit(
        df_y=prices.iloc[:-1], windows=[5], lags=[1], fit_func=lambda *args: None
    )

    simulate_module.simulate(
        sims=1,
        df_y=prices,
        days_out=1,
        windows=[5],
        lags=[1],
        sim_func=lambda sims, df_y, *args: [df_y.tail(1).assign(simulation=0)],
    )
