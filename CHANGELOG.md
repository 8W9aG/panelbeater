# Changelog

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
