# Changelog

## 0.3.1

### Fixed: `simulate()` would fail once the copula is refitted

`fit()` fitted the vine copula on returns (`df_y.pct_change().dropna()`), but
`simulate()` loaded it with prices. The cache file is found by column names, so the
right file was opened, but `load_vine` then validates its contents against the frame
it is given and raises `ValueError: historical rows changed since the fit`.

**Impact.** None so far: the cache files written before 0.3.0 have no metadata
sidecar, so the check was skipped. The first training run with 0.3.0 or later writes
a sidecar, and inference after that would have failed.

Both sides now build the frame with the new `copula.vine_returns(df_y, horizon=1)`
(`fit()` for every horizon, `simulate()` when loading).

### Changed: simulation advances paths in lockstep batches

`simulate()` used to run one task per path. Every path built its own wavetrainer
`Trainer` and called `transform()` on a single row each simulated day. That call's
cost is almost all fixed per trained column, not per row: on ~480 trained columns, one
row took ~102s and six stacked rows ~109s.

Now `simulate()` splits the paths into one batch per worker. The new
`run_simulation_batch()` advances a batch's paths together, one day at a time, and
makes **one** `transform()` call per day for the whole batch, with one `Trainer`
shared by every path in it. This is exact because every path visits the same dates:
`denormalize` always appends the next calendar day, and wavetrainer picks each row's
walk-forward fold from its date alone. A stacked call was verified to return
bit-identical predictions to per-row calls on the real model folder.

Measured at small scale (12 paths × 2 days, 2 workers): 2,448s before, 760s after
(3.2x). The gain grows with batch size, because the `transform()` cost is paid once
per batch per day. `features()` is still computed per path (~31s per path per day)
and is now most of the remaining time.

**Reproducibility.** A single path is unchanged: with `np.random` and the vine seeded,
`run_single_simulation` gives bit-identical output to 0.3.0. With several paths, each
day's copula draws are now taken path by path within the batch, so the draws
interleave differently and a seeded multi-path run differs from 0.3.0's. The
distributions are the same: comparing per-asset means and spreads of simulated
returns between the two loops showed no difference beyond sampling noise.

### Fixed: nested parallelism in simulation

`run_single_simulation` called `Trainer.transform()` with wavetrainer's default
`n_jobs=-1` from inside `simulate()`'s own `Parallel(n_jobs=-1)`, so each worker
started a second pool: up to cores² processes. On a 10-core, 64GB machine swap grew
by 33GB within three minutes. Simulation now
calls `transform()` with `n_jobs=1`.

### Added

- `run_simulation_batch(sim_indices, df_y, days_out, windows, lags, vine_fit, ...,
  n_jobs=1)`, returning one frame per path in the same form `run_single_simulation`
  returns.
- `simulate(..., n_jobs=-1)`: how many batches run at once. Each worker holds every
  trained column's models (~5GB) and `features()` briefly needs ~13GB more, so a
  worker peaks at ~15-18GB. Set it from memory, not core count. The CLI now uses 3.
- `run_single_simulation(..., n_jobs=1)`. It is now a batch of one, and `simulate()`
  no longer calls it. A caller that runs one path per machine (e.g. one per
  container) should pass `n_jobs=-1`: at the default, one path runs `transform()`
  on one core (~3.7x slower than with every core, measured on wavetrainer 0.3.26).
- `copula.vine_returns(df_y, horizon=1)`.
- `run.sh` reruns `panelbeater` until it exits 0, with backoff, and stops on a
  deliberate signal (Ctrl-C, SIGTERM…) but retries crashes. Its watchdog treats a
  run with no output for `HANG_TIMEOUT` seconds (default 1200) as hung, dumps every
  thread's stack with `py-spy` to `hangs/`, kills it and retries. Settings are
  environment variables, documented at the top of the script. `py-spy` is added to
  `requirements-dev.txt`, and `hangs/` is gitignored.
- Tests: `test_simulate.py` (batching logic), `test_fit_simulate.py` (`fit()` then
  `simulate()` share the vine) and `test_denormalize_snapshot.py`.

### Changed

- **Requires wavetrainer >= 0.3.27.** On pandas 3, earlier versions fail every
  `transform()` call ("assignment destination is read-only" in the power-transformer
  normaliser). 0.3.27 makes that normaliser a no-op, which is what it already was on
  pandas 2, so predictions are unchanged (verified bit-identical on the real model
  folder), and makes `transform()` about 3x faster.
- `denormalize` is faster: `y`'s columns are added in one step, each series' columns
  are grouped once instead of scanned for every lookup, and the new row is written in
  one assignment. Output is unchanged: `test_denormalize_snapshot.py` checks seeded
  results bit for bit against the previous implementation, including how many random
  numbers each call consumes.
- Training runs Optuna trials with `n_jobs=1` (was 4). Optuna runs parallel trials as
  threads in one process, and concurrent CatBoost fits there segfault and deadlock.
  Each fit already uses every core.

## 0.3.0

### Fixed: copula uniforms were assigned to the wrong columns (joint quantities affected)

`denormalize` consumed the copula's joint uniforms **positionally** while iterating a
`set` of column names. Set iteration order is unrelated to the copula's column order
and is salted per process by `PYTHONHASHSEED`, so each worker process assigned the
uniforms to a different permutation of series.

**Impact.** Per-series marginals were unaffected — every series still received *a*
uniform drawn from the right distribution. But the copula's **dependence structure was
randomly permuted across columns**, so any *joint* quantity computed from simulations
is wrong: correlations between series, and anything derived from two or more series
together (combined totals, margins, head-to-head outcomes, portfolio aggregates).

**Simulations produced by earlier versions need to be re-run.** Refit is not required
for this bug — only re-simulation — but see the cache fix below.

The z-score reconstruction path had the same defect as the quantile path; both are
fixed. Series are now iterated in sorted order and each looks up its own uniform by
name.

#### Breaking API change

`denormalize` no longer accepts a bare, unlabelled `u_sample` array, because there is
no safe way to infer which uniform belongs to which series. Pass either:

```python
from panelbeater.copula import load_vine

vine = load_vine(df_returns=returns)      # VineFit: copula + fitted column order
u = vine.sample()                         # pd.Series indexed by series name
denormalize(df, y=panel, u_sample=u, quantiles=5)
```

or, to keep an array:

```python
denormalize(df, y=panel, u_sample=arr, u_columns=["SCORE_X", "NETYARDS_X", ...])
```

A series present in `y` but absent from the uniforms is sampled **independently** and
raises a `UserWarning` naming the affected series. It never consumes another series'
uniform.

### Fixed: vine copula cache key ignored the data and had no namespace

The cache key hashed only the column names and horizon, so:

- appending rows or correcting history produced the same key, and a refit could
  silently reuse a stale fit;
- two datasets sharing column names collided on one file;
- `load_vine_copula` ignored the TTL that `fit_vine_copula` checked.

Now:

- `cache_dir` and `namespace` arguments control where the cache lives and how it is
  keyed (default remains the current working directory, unnamespaced);
- a JSON sidecar beside each pickle records the **ordered fitted columns**, row count,
  first/last index, a content hash, horizon, namespace and fit timestamp;
- `fit_vine` refits when the columns or the data contents change;
- `load_vine` reuses a fit when the current data is the fitted data **plus appended
  rows only**, logging what it reused, and raises clearly when historical rows changed
  or rows were removed — it will not return a fit for data that no longer exists;
- writes are atomic (temp file + `os.replace`), safe for concurrent worker reads.

Cache files written before 0.3.0 have no sidecar; they still load, with a warning that
they cannot be validated. Refit to clear it.

### Added

- `VineFit` (`copula` + fitted `columns`, with `.sample()` returning labelled uniforms),
  plus `fit_vine` / `load_vine`. The older `fit_vine_copula` / `load_vine_copula`
  remain and return just the copula.
- `sample_joint_step(cop, columns=...)` returns a labelled `pd.Series`.
- `num_threads` on `fit_vine`, defaulting to every core. Verified that threading leaves
  the fitted model bit-identical — structure, families and parameters all match a
  single-threaded fit — so this is a pure speedup. A 96-dimension fit previously ran
  single-threaded.
