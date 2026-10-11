"""Tests for the lockstep batched simulation loop.

The trainer, features, denormalize and the copula are replaced by small
deterministic fakes, so these check the loop's bookkeeping (which row goes to
which path, draw order, batching) rather than the models.
"""

import importlib

import pandas as pd
import pytest

from panelbeater.copula import vine_returns

# The package re-exports a function with the same name as this module.
sim = importlib.import_module("panelbeater.simulate")


class FakeVine:  # pylint: disable=too-few-public-methods
    """Hands out 1, 2, 3, ... so the draw order is visible in the results."""

    def __init__(self):
        self.draws = 0

    def sample(self):
        """The next uniform, as a plain number."""
        self.draws += 1
        return float(self.draws)


class FakeTrainer:  # pylint: disable=too-few-public-methods
    """Predicts 1% of each row's feature, and records every transform() call."""

    def __init__(self):
        self.calls = []

    def transform(self, df, ignore_no_dates=False, n_jobs=-1):
        """Adds a prediction column, as wavetrainer does."""
        assert ignore_no_dates
        self.calls.append({"index": df.index, "n_jobs": n_jobs})
        out = df.copy()
        out["pred"] = df["feat"] * 0.01
        return out


def fake_features(df, **kwargs):
    """One feature: the price itself."""
    del kwargs
    return pd.DataFrame({"feat": df["A"]}, index=df.index)


def fake_denormalize(df, y, u_sample, quantiles, precomputed_kdes):
    """Next price = last + prediction + draw, appended one day later."""
    del quantiles, precomputed_kdes
    assert len(df) == 1
    assert df.index[0] == y.index[-1]  # the prediction is for this path's last day
    next_value = y["A"].iloc[-1] + df["pred"].iloc[0] + u_sample
    next_row = pd.DataFrame(
        {"A": [next_value]}, index=[y.index[-1] + pd.Timedelta(days=1)]
    )
    return pd.concat([y, next_row])


@pytest.fixture(name="df_y")
def df_y_fixture():
    """Five days of one price series, ending at 100."""
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    return pd.DataFrame({"A": [96.0, 97.0, 98.0, 99.0, 100.0]}, index=idx)


@pytest.fixture(name="trainers")
def trainers_fixture(monkeypatch):
    """Patches the simulation's dependencies; yields every Trainer created."""
    created = []

    def create_wt():
        trainer = FakeTrainer()
        created.append(trainer)
        return trainer

    monkeypatch.setattr(sim, "create_wt", create_wt)
    monkeypatch.setattr(sim, "features", fake_features)
    monkeypatch.setattr(sim, "denormalize", fake_denormalize)
    return created


def run_batch(df_y, sim_indices, days_out, **kwargs):
    """run_simulation_batch with the fixed arguments filled in."""
    return sim.run_simulation_batch(
        sim_indices, df_y, days_out, [5], [1], FakeVine(), **kwargs
    )


def test_one_stacked_transform_per_day(df_y, trainers):
    """A batch shares one Trainer and makes one transform() call per day."""
    run_batch(df_y, [0, 1, 2], days_out=2)

    assert len(trainers) == 1
    calls = trainers[0].calls
    assert len(calls) == 2
    for day, call in enumerate(calls, start=1):
        assert len(call["index"]) == 3  # one row per path
        # every path's row is for the same date: the batch is in lockstep
        assert set(call["index"]) == {df_y.index[-1] + pd.Timedelta(days=day - 1)}


def test_each_path_gets_its_own_prediction_and_draws_in_order(df_y, trainers):
    """Row i of the stacked prediction goes to path i, draws go path by path."""
    del trainers
    results = run_batch(df_y, [0, 1], days_out=2)

    # Draws are taken day by day, path by path: day 1 gets 1 and 2, day 2 gets
    # 3 and 4. Each step is last * 1.01 + draw.
    path_0, path_1 = (r["A"].iloc[-2:].tolist() for r in results)
    assert path_0 == pytest.approx([100 * 1.01 + 1, 102 * 1.01 + 3])  # 102, 106.02
    assert path_1 == pytest.approx([100 * 1.01 + 2, 103 * 1.01 + 4])  # 103, 108.03
    # On day 2 the paths' features differ (102 vs 103), so swapping the
    # prediction rows would give 106.03 and 108.02 instead.


def test_results_are_tails_labelled_by_path(df_y, trainers):
    """Each path returns days_out + 1 rows, marked with its own index."""
    del trainers
    results = run_batch(df_y, [7, 3], days_out=2)

    assert len(results) == 2
    for sim_idx, result in zip([7, 3], results):
        assert len(result) == 3
        assert result.index[0] == df_y.index[-1]  # starts from the last real day
        assert (result[sim.SIMULATION_COLUMN] == sim_idx).all()
    assert results[0].index.equals(results[1].index)  # same dates for every path


def test_save_state_func_sees_every_path_every_day(df_y, trainers):
    """save_state_func is called per path per day, with that path's frame."""
    del trainers
    saved = []
    run_batch(
        df_y,
        [4, 5],
        days_out=2,
        save_state_func=lambda i, frame, final: saved.append((i, len(frame), final)),
    )

    assert saved == [(4, 6, False), (5, 6, False), (4, 7, False), (5, 7, False)]


def test_callers_frame_is_not_modified(df_y, trainers):
    """The paths work on their own frames, not the one passed in."""
    del trainers
    before = df_y.copy()
    run_batch(df_y, [0, 1], days_out=2)
    pd.testing.assert_frame_equal(df_y, before)


def test_n_jobs_reaches_transform(df_y, trainers):
    """n_jobs controls transform()'s column parallelism, and defaults to 1."""
    run_batch(df_y, [0], days_out=1)
    run_batch(df_y, [0], days_out=1, n_jobs=-1)
    assert [t.calls[0]["n_jobs"] for t in trainers] == [1, -1]


def test_single_simulation_is_a_batch_of_one(df_y, trainers):
    """run_single_simulation returns what a one-path batch does."""
    single = sim.run_single_simulation(9, df_y, 2, [5], [1], FakeVine(), n_jobs=-1)
    batch = run_batch(df_y, [9], days_out=2)[0]

    pd.testing.assert_frame_equal(single, batch)
    assert single["A"].iloc[-2:].tolist() == pytest.approx(
        [100 * 1.01 + 1, 102 * 1.01 + 2]
    )
    assert trainers[0].calls[0]["n_jobs"] == -1


@pytest.fixture(name="simulate_env")
def simulate_env_fixture(df_y, tmp_path, monkeypatch):
    """Runs simulate() in-process, recording the batches and n_jobs it uses."""
    monkeypatch.chdir(tmp_path)  # simulate() writes sims.parquet to the cwd
    monkeypatch.setattr(sim, "load_vine", lambda df_returns: FakeVine())
    record = {"batches": [], "n_jobs": []}

    def fake_batch(sim_indices, frame, days_out, *args):
        del args
        record["batches"].append(list(sim_indices))
        return [
            frame.tail(days_out + 1).assign(**{sim.SIMULATION_COLUMN: i})
            for i in sim_indices
        ]

    class SequentialParallel:  # pylint: disable=too-few-public-methods
        """Stands in for joblib.Parallel, running the tasks here, in order.

        The real one runs batches in worker processes, which would not see
        this test's monkeypatches.
        """

        def __init__(self, n_jobs):
            record["n_jobs"].append(n_jobs)

        def __call__(self, tasks):
            return [func(*args, **kwargs) for func, args, kwargs in tasks]

    monkeypatch.setattr(sim, "run_simulation_batch", fake_batch)
    monkeypatch.setattr(sim, "Parallel", SequentialParallel)
    # A 4-core machine: -1 means every core, otherwise the count asked for.
    monkeypatch.setattr(
        sim.joblib, "effective_n_jobs", lambda n_jobs: 4 if n_jobs == -1 else n_jobs
    )
    return df_y, record


def test_simulate_splits_paths_into_one_batch_per_worker(simulate_env):
    """10 paths over 4 workers become 4 batches, covering every path once."""
    df_y, record = simulate_env
    df_mc = sim.simulate(sims=10, df_y=df_y, days_out=1, windows=[5], lags=[1])

    assert record["batches"] == [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    assert df_mc[sim.SIMULATION_COLUMN].tolist() == [
        i for i in range(10) for _ in range(2)
    ]  # flattened in path order, two rows each


def test_simulate_never_makes_empty_batches(simulate_env):
    """Fewer paths than workers gives one batch per path."""
    df_y, record = simulate_env
    sim.simulate(sims=2, df_y=df_y, days_out=1, windows=[5], lags=[1])
    assert record["batches"] == [[0], [1]]


def test_simulate_n_jobs_sets_workers_and_batches(simulate_env):
    """simulate(n_jobs=...) sets how many batches there are and run at once."""
    df_y, record = simulate_env
    sim.simulate(sims=6, df_y=df_y, days_out=1, windows=[5], lags=[1], n_jobs=2)
    assert record["n_jobs"] == [2]
    assert record["batches"] == [[0, 1, 2], [3, 4, 5]]

    record["batches"].clear()
    sim.simulate(sims=6, df_y=df_y, days_out=1, windows=[5], lags=[1])
    assert record["n_jobs"] == [2, -1]
    assert record["batches"] == [[0, 1], [2, 3], [4], [5]]


def test_simulate_sim_func_bypasses_batching(simulate_env):
    """A custom sim_func is called as before, and no batches are run."""
    df_y, record = simulate_env
    calls = []

    def sim_func(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        sims, frame, days_out, windows, lags, vine_fit
    ):
        calls.append((sims, days_out, windows, lags, type(vine_fit)))
        return [frame.tail(1).assign(**{sim.SIMULATION_COLUMN: 0})]

    sim.simulate(
        sims=5, df_y=df_y, days_out=2, windows=[5], lags=[1], sim_func=sim_func
    )
    assert calls == [(5, 2, [5], [1], FakeVine)]
    assert not record["batches"]


def test_vine_returns_matches_pct_change():
    """vine_returns is the horizon's percentage change, leading NaN rows dropped."""
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 99.0, 99.0], "B": [None, 50.0, 55.0, 44.0]},
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )
    one_day = vine_returns(prices)
    assert one_day.index.equals(prices.index[2:])  # B's first return is NaN too
    assert one_day.loc[prices.index[2]].tolist() == pytest.approx([-0.1, 0.1])

    two_day = vine_returns(prices, horizon=2)
    assert two_day.index.equals(prices.index[3:])
    assert two_day.loc[prices.index[3]].tolist() == pytest.approx([-0.1, -0.12])


def test_vine_returns_accepts_a_series():
    """A single series comes back as a one-column frame."""
    prices = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
        name="A",
    )
    returns = vine_returns(prices)  # type: ignore[arg-type]
    assert isinstance(returns, pd.DataFrame)
    assert list(returns.columns) == ["A"]
