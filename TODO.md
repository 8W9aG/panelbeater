# TODO

Roughly in priority order. Items marked **(verified)** were reproduced, not just read.

## Bugs

- [x] **`simulate()` will fail as soon as the copula is refitted (verified).**
  Fixed: `copula.vine_returns()` builds the frame for `fit()` (every horizon)
  and `simulate()`, tested in `tests/test_fit_simulate.py`.
  `fit()` fits the vine on returns (`df_y.pct_change().dropna()`), but
  `simulate()` calls `load_vine(df_returns=df_y)` with *prices*
  ([simulate.py](panelbeater/simulate.py), `simulate`). The cache key only uses
  column names, so the same file is found, but the content-hash check in
  `_describe_reuse` then raises `ValueError: historical rows changed since the fit`.
  It works today only because the `market_structure_*.pkl` files on disk predate
  the metadata sidecar, so the check is skipped. The next training run writes a
  sidecar, and inference after that will break. Fix: pass
  `df_y.pct_change().dropna()` in `simulate()`, or have one helper build the
  returns frame for both callers. Add a test that runs `fit` and then `simulate`
  on the same data.

- [ ] **FRED retry never retries.** In `_fetch_fred_with_retry`
  ([download.py](panelbeater/download.py)), the `return None` sits inside the
  `except` block. The first failure sleeps and then returns `None` instead of
  looping. The caller turns that into `ValueError("Could not download FRED series")`,
  which falls through to the `get_series` fallback (and that has lookahead bias).
  Remove the `return None`, or move it after the loop.

- [ ] **Possible one-day lag at inference.** `features()` ends with `.shift(1)`,
  so the row at date T is built from data up to T-1 and predicts the T-1→T return.
  In `run_simulation_batch`, `features(path_y).iloc[[-1]]` is that T row, but its
  prediction is used for the new row T+1. The latest observed day never reaches
  the model. Check this, and if it's right, add a placeholder T+1 row before
  featurising (or skip the shift for the last row).

- [ ] **panelbeater-modal's simulation is broken (verified).** Its `sim_func`
  calls `run_single_simulation(..., vine_cop=...)`
  ([train.py](../panelbeater-modal/train.py)), but the parameter is `vine_fit`,
  so every call raises `TypeError`. When fixing it, also pass `n_jobs=-1`:
  `run_single_simulation` now defaults to `transform(n_jobs=1)`, and one path
  per container measured ~3.7x slower that way. See the Performance item on
  batching Modal.

- [ ] **Inference needs wavetrainer 0.3.27, which isn't published yet.** On
  pandas 3, older wavetrainer's power-transformer normaliser raises
  "assignment destination is read-only" on every `transform()` call (including
  at the last commit). 0.3.27 makes that normaliser a no-op, with identical
  predictions. `requirements.txt` already pins `wavetrainer>=0.3.27`, but it is
  only built locally (`~/dev/wavetrain/dist/`), so a fresh install, including
  Modal's, fails to resolve until it's published.

## Performance

Measured 2026-10-11 on the 10-core, 64GB machine with ~480 trained columns.

- [ ] **`features()` is now the simulation bottleneck (verified).** Each call
  takes ~31s and briefly needs ~13GB, and simulation calls it once per path per
  day. It builds every one of the ~70,700 feature columns over ~2,100 rows, and
  then only the last row is used. After batching, `transform()` costs ~110s per
  *batch* per day, so features is almost all of the per-path cost at
  production batch sizes (~99% with 1,000 paths over 3 workers). Options: compute only the last row, or update it incrementally from
  the previous day; skip columns that no trained model's selector keeps.
  Fixing the memory peak also lets `_SIM_N_JOBS` go up (next item).
- [ ] **`_SIM_N_JOBS = 3` is a memory limit, not a CPU one.** Each simulation
  worker peaks at ~15-18GB (~5GB of cached models plus the `features()` peak),
  so one per core would need ~150GB. Revisit after the `features()` item, and
  if the model set grows.
- [ ] **Batch paths on Modal.** Modal runs one path per container through
  `run_single_simulation`, so it pays the full ~100s `transform()` every day
  for every path. Running ~100 paths per container through
  `run_simulation_batch` shares that across the batch. Its resume-from-parquet
  logic would need to become per path (`save_state_func` already gets
  `sim_idx`).
- [ ] **Time a full-size run on an idle machine.** Batching was only timed at
  12 paths × 2 days with 2 workers, with ATP training running alongside: old
  loop 2,448s, new 760s (3.2x). The gain should grow with batch size, since
  `transform()` cost is per batch. Time `_SIMS`/`_DAYS_OUT` as configured.

## Modelling

- [ ] **Percent change is ill-defined for rate and spread series.** `T10Y2Y`
  crosses zero, and `FEDFUNDS` and `^IRX` have sat near zero. `pct_change`
  blows up on these series, and simulating `y * (1 + value)` from a negative base
  flips the sign. Use first differences (and additive denormalisation) for
  level-type macros: rates, spreads, `UNRATE`, `VIXCLS`?
- [ ] **`GDP` as a flagged quantile target.** GDP is quarterly and forward-filled
  to daily, so nearly every 1-day change is exactly 0. The 365-day rolling rank
  then puts nearly everything in one bucket, and the KDEs are degenerate. Either
  drop it from `_FLAGGED_MACROS` or model it at release frequency.
- [ ] **Z-score tails are truncated.** Targets are clipped to ±3σ, and
  `denormalize` samples the ±3 buckets as `3 ± 0.25`σ. Every non-flagged series
  therefore has no moves beyond ~3.25σ, which matters a lot when pricing OTM
  options. Use the same tail-KDE approach as the flagged series for the edge
  buckets.
- [ ] **FRED release calendar only covers the last 100 releases.**
  `fetch_fred_release_dates(limit=100, sort_order=desc)`: for daily or weekly
  releases (DGS10, VIXCLS…) that is only a few months, so `days_since/until_macro`
  is NaN for most of the training history but populated at inference time. That
  is a train/serve skew. Page through the full history, or drop the feature for
  high-frequency releases.
- [ ] **Fallback FRED series have lookahead bias.** When
  `get_series_all_releases` fails, `get_series` is indexed by observation date.
  At minimum, shift it by a per-series publication lag. Also log loudly (or fail
  the run) rather than only printing to `tqdm.write`.
- [ ] **Simulation steps are calendar days, not trading days.** `denormalize`
  appends `index[-1] + 1 day`, and `_DAYS_OUT = 32` counts calendar days. This
  is consistent with training (weekends are forward-filled), but the
  option-expiry horizon has to be mapped onto calendar days explicitly, and the
  `_null` probability carries the weekend effect. Document this, or switch to a
  business-day index.
- [ ] **Don't model a series on days it can't move.** At the moment every series
  is predicted and sampled every calendar day, and the model has to learn
  "no change" through the `_null` probability. When the calendar already
  guarantees a zero move, skip the model and set the move to exactly 0 instead:
  - **NYSE/Nasdaq tickers** (SPY, QQQ, NVDA…): weekends and exchange holidays.
    Use `exchange_calendars` or `pandas_market_calendars` (XNYS).
  - **Other venues** each have their own calendar: CME for futures (`GC=F`,
    `CL=F`…), the Saturday close for FX, and JPX for `^N225`. Crypto trades
    every day, so it never skips.
  - **Low-frequency FRED macros** (GDP, UNRATE, CPIAUCSL, M2SL, INDPRO…): the
    value only changes on a release date. Reuse `fetch_fred_release_dates`, which
    already returns scheduled future releases, so any day that isn't a release
    is a guaranteed zero.

  Apply the same mask in training. In `normalize`, set targets on guaranteed-zero
  days to NaN, so the model learns the distribution of moves on days that can
  actually move, rather than a mix that is mostly forced zeros. This should
  improve calibration, particularly for the quantile buckets (see the GDP item
  above). In simulation, this also saves compute: only transform the columns
  that are live that day (wavetrainer's `transform` takes a `columns=` subset
  since 0.3.25), and on weekends only crypto and the daily FRED series are
  live.
- [ ] The copula is fitted on `pct_change().dropna()` across *all* columns, so
  it only sees the period where every series exists (BITO starts late 2021, so
  roughly 4 years). Consider pairwise or complete-case alternatives, or say in
  the docs that this is deliberate.

## Product and README

- [ ] The README describes steps 4–6 (fetch option chains, find mispriced
  options, Kelly sizing), but none of that is in the package. `trades.csv` and
  the gitignored `generate_model_report.py` / `dependencies.py` suggest it exists
  locally. Either bring it into `panelbeater/`, or update the README to say the
  CLI stops at writing `sims.parquet`.
- [ ] `--inference` defaults to `False`, so plain `panelbeater` does not do
  "a full train, inference and attempts to find mispriced options" as the README
  says.
- [ ] The README dependency list is stale. It is missing fredapi, statsmodels,
  pyarrow and tabpfn-extensions, and lists pandas-datareader, which nothing
  imports. It also doesn't mention that `FRED_API_KEY` is required (in `.env`).
- [ ] The `requests_cache.CachedSession` in `main()` is passed to `download()`
  but never used: yfinance gets `session=None`, and fredapi uses its own HTTP.
  Wire it up or remove it (along with the `panelbeater-cache.sqlite` it creates).

## Code health

- [ ] Commit the current working-tree changes (denormalize/simulate perf work,
  batched simulation, wavetrainer 0.3.27 bump, run.sh watchdog, `n_jobs=1`)
  and the untracked snapshot test, and add a CHANGELOG entry.
- [ ] **Test simulation against a real (small) trained model.**
  `tests/test_simulate.py` covers the batching logic with fakes for the
  trainer, features, denormalize and vine. Exactness was only checked with
  one-off scripts against the real model folder. Train a tiny Trainer in a
  test and assert that a stacked `transform()` equals per-row calls, and that
  with seeded `np.random` and vine `run_single_simulation` equals the old
  per-path loop.
- [ ] `tabpfn-extensions` is probably no longer needed in `requirements.txt`.
  Its comment says wavetrainer imports it, but wavetrainer dropped TabPFN in
  0.3.26.
- [ ] Add CI (GitHub Actions) running `lint.sh` and `test.sh`. There is
  currently no `.github/`.
- [ ] Lint `tests/` too. `lint.sh` only covers `panelbeater/`.
- [ ] Move config out of `__main__.py` (tickers, macros, flagged lists, windows,
  lags, sims, days out) into a config file or CLI options, so experiments don't
  need code edits.
- [ ] The `horizons` branch of `fit()` isn't reachable from the CLI, and
  `denormalize` doesn't understand the `_h{h}` targets it produces. Either finish
  it or delete it. Also remove the `# --- NEW ---` / `# --- UPDATED ---` comment
  markers.
- [ ] Drop or deprecate the `load_vine_copula` / `fit_vine_copula` wrappers once
  nothing uses them.
- [ ] Packaging: move to `pyproject.toml` and keep a single version source
  (it's currently duplicated in `setup.py` and `__init__.py`). Question why
  `setuptools<81` is a runtime requirement.
- [ ] Stop tracking `trades.csv` (or move it to `examples/`). It has two
  redundant leading index columns from a `to_csv` without `index=False`.
- [ ] `yf.download(proxy=...)` is deprecated in recent yfinance. Remove the
  argument.
