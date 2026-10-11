"""Handle joint distributions."""

# pylint: disable=too-many-locals,pointless-string-statement
import datetime
import hashlib
import json
import os
import pickle
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast, overload

import numpy as np
import pandas as pd
import pyvinecopulib as pv

_METADATA_VERSION = 1


@dataclass(frozen=True)
class VineFit:
    """
    A fitted vine copula bundled with the column order it was fitted on.

    The column order matters: `Vinecop.simulate` returns uniforms positionally,
    so a caller that maps them onto series by anything other than this exact
    order silently permutes the dependence structure.
    """

    copula: pv.Vinecop
    columns: list[str]

    def sample(self) -> pd.Series:
        """One joint uniform sample, labelled with the fitted column names."""
        return sample_joint_step(self.copula, columns=self.columns)


def _frame_hash(df_returns: pd.DataFrame) -> str:
    """Content hash of a frame: column names, values and index."""
    digest = hashlib.md5()
    digest.update("\x1f".join(map(str, df_returns.columns)).encode("utf-8"))
    digest.update(np.ascontiguousarray(df_returns.to_numpy(dtype=float)).tobytes())
    digest.update("\x1f".join(map(str, df_returns.index)).encode("utf-8"))
    return digest.hexdigest()


def _vine_paths(
    df_returns: pd.DataFrame,
    horizon: int,
    cache_dir: str | None,
    namespace: str | None,
) -> tuple[str, str]:
    """
    Returns the (pickle, metadata) paths for this fit.

    The key covers the columns, the horizon and the namespace, so two datasets
    that happen to share column names (two sports with the same team names, say)
    no longer collide on one file.
    """
    key = "-".join(sorted(df_returns.columns.values.tolist()))
    if namespace:
        key = f"{namespace}\x1f{key}"
    struct_str = hashlib.md5(key.encode("utf-8")).hexdigest()
    stem = f"market_structure_{struct_str}_h{horizon}"
    directory = cache_dir if cache_dir is not None else "."
    return os.path.join(directory, f"{stem}.pkl"), os.path.join(
        directory, f"{stem}.json"
    )


def _write_atomic(path: str, data: bytes) -> None:
    """Write via a temp file + os.replace, so concurrent readers never see a partial file."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle_fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _read_metadata(meta_file: str) -> dict | None:
    """Reads the sidecar, or None when it is missing/unreadable."""
    if not os.path.exists(meta_file):
        return None
    try:
        with open(meta_file, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict) or meta.get("version") != _METADATA_VERSION:
        return None
    return meta


def _describe_reuse(df_returns: pd.DataFrame, meta: dict) -> str | None:
    """
    Decides whether `meta` still describes `df_returns`.

    Returns None when the stored fit is reusable, otherwise a human-readable
    reason why it is not. Appending rows is fine -- the stored fit was computed
    on a prefix of the current data, which still exists. Changing or removing
    historical rows is not.
    """
    columns = sorted(df_returns.columns.values.tolist())
    if meta.get("columns") != columns:
        return "the column set changed"

    n_rows = meta.get("n_rows")
    if not isinstance(n_rows, int):
        return "the stored row count is unusable"
    if len(df_returns) < n_rows:
        return f"rows were removed ({len(df_returns)} now, {n_rows} at fit time)"

    if _frame_hash(df_returns.iloc[:n_rows]) != meta.get("content_hash"):
        return "historical rows changed since the fit"

    return None


def _metadata(
    df_returns: pd.DataFrame, horizon: int, namespace: str | None
) -> dict[str, Any]:
    """Sidecar describing exactly what a fit was computed on."""
    return {
        "version": _METADATA_VERSION,
        "columns": sorted(df_returns.columns.values.tolist()),
        "n_rows": len(df_returns),
        "first_index": str(df_returns.index[0]) if len(df_returns) else None,
        "last_index": str(df_returns.index[-1]) if len(df_returns) else None,
        "content_hash": _frame_hash(df_returns),
        "horizon": horizon,
        "namespace": namespace,
        "fitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def load_vine(
    df_returns: pd.DataFrame,
    horizon: int = 1,
    cache_dir: str | None = None,
    namespace: str | None = None,
) -> VineFit:
    """
    Loads a cached vine copula along with the columns it was fitted on.

    Raises FileNotFoundError when nothing is cached, and ValueError when the
    cached fit no longer describes `df_returns` -- better than silently
    simulating from a model of data that no longer exists.
    """
    df_returns = df_returns.reindex(sorted(df_returns.columns), axis=1)
    vine_file, meta_file = _vine_paths(df_returns, horizon, cache_dir, namespace)

    with open(vine_file, "rb") as f:
        cop = pickle.load(f)

    meta = _read_metadata(meta_file)
    if meta is None:
        # Pre-sidecar cache file: we cannot verify what it was fitted on.
        print(
            f"Warning: {vine_file} has no metadata sidecar, so it cannot be "
            "validated against the current data. Refit to silence this."
        )
        return VineFit(copula=cop, columns=sorted(df_returns.columns.values.tolist()))

    reason = _describe_reuse(df_returns, meta)
    if reason is not None:
        raise ValueError(
            f"Cached vine copula at {vine_file} cannot be reused because "
            f"{reason}. Refit with fit_vine()."
        )

    appended = len(df_returns) - int(meta["n_rows"])
    suffix = f" plus {appended} appended row(s)" if appended else ""
    print(
        f"Reusing vine copula (horizon={horizon}) from {vine_file}: "
        f"{meta['n_rows']} rows at fit time{suffix}."
    )
    return VineFit(copula=cop, columns=list(meta["columns"]))


def vine_returns(df_y: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    The returns frame a price panel's vine is fitted on and loaded with.

    The cache is found by column names but validated against these exact
    values, so fitting and loading must both build it here: loading with the
    prices instead finds the right file and then fails the content check.
    """
    returns = df_y.pct_change(periods=horizon).dropna()
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    return returns


def fit_vine(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    df_returns: pd.DataFrame,
    horizon: int = 1,
    ttl_days: int = 30,
    cache_dir: str | None = None,
    namespace: str | None = None,
    num_threads: int | None = None,
) -> VineFit:
    """
    Returns a fitted vine copula and the column order it was fitted on.

    Reuses a cached fit when it is within `ttl_days` and was fitted on exactly
    this data; refits when the columns or the data contents changed.

    `num_threads` defaults to every core. Threading does not change the fitted
    model -- structure, families and parameters are identical to a
    single-threaded fit -- it only makes the fit faster.
    """
    df_returns = df_returns.reindex(sorted(df_returns.columns), axis=1)
    vine_file, meta_file = _vine_paths(df_returns, horizon, cache_dir, namespace)

    # 1. Check for a valid cached model
    if os.path.exists(vine_file):
        meta = _read_metadata(meta_file)
        file_age_seconds = time.time() - os.path.getmtime(vine_file)
        expired = file_age_seconds >= (ttl_days * 24 * 60 * 60)
        reason = "the cache expired" if expired else None
        if reason is None:
            reason = (
                _describe_reuse(df_returns, meta)
                if meta is not None
                else "it has no metadata sidecar"
            )
        if reason is None:
            print(f"Loading cached vine copula (horizon={horizon}) from {vine_file}")
            with open(vine_file, "rb") as f:
                return VineFit(
                    copula=pickle.load(f), columns=list(cast(dict, meta)["columns"])
                )
        print(f"Refitting vine copula (horizon={horizon}): {reason}.")
    else:
        print(f"Vine copula (horizon={horizon}) is missing. Fitting new model...")

    # 2. Fit a new one
    n = len(df_returns)
    # Manual PIT transform to Uniform [0, 1]
    u = df_returns.rank(method="average").values / (n + 1)

    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student],  # type: ignore
        tree_criterion="tau",
        num_threads=num_threads if num_threads is not None else (os.cpu_count() or 1),
    )

    cop = pv.Vinecop.from_data(u, controls=controls)

    # 3. Save the model and its sidecar, atomically
    _write_atomic(vine_file, pickle.dumps(cop, protocol=pickle.HIGHEST_PROTOCOL))
    meta = _metadata(df_returns, horizon, namespace)
    _write_atomic(meta_file, json.dumps(meta, indent=2).encode("utf-8"))

    return VineFit(copula=cop, columns=list(meta["columns"]))


def load_vine_copula(
    df_returns: pd.DataFrame,
    horizon: int = 1,
    cache_dir: str | None = None,
    namespace: str | None = None,
) -> pv.Vinecop:
    """Loads a vine copula model for a specific horizon.

    Prefer `load_vine`, which also returns the fitted column order needed to
    label samples correctly.
    """
    return load_vine(
        df_returns=df_returns,
        horizon=horizon,
        cache_dir=cache_dir,
        namespace=namespace,
    ).copula


def fit_vine_copula(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    df_returns: pd.DataFrame,
    horizon: int = 1,
    ttl_days: int = 30,
    cache_dir: str | None = None,
    namespace: str | None = None,
    num_threads: int | None = None,
) -> pv.Vinecop:
    """Returns a fitted vine copula.

    Prefer `fit_vine`, which also returns the fitted column order needed to
    label samples correctly.
    """
    return fit_vine(
        df_returns=df_returns,
        horizon=horizon,
        ttl_days=ttl_days,
        cache_dir=cache_dir,
        namespace=namespace,
        num_threads=num_threads,
    ).copula


@overload
def sample_joint_step(
    cop: pv.Vinecop, columns: None = None
) -> np.ndarray[Any, np.dtype[np.float64]]: ...


@overload
def sample_joint_step(cop: pv.Vinecop, columns: Sequence[str]) -> pd.Series: ...


def sample_joint_step(
    cop: pv.Vinecop, columns: Sequence[str] | None = None
) -> np.ndarray[Any, np.dtype[np.float64]] | pd.Series:
    """
    Returns one joint sample vector for the panel.

    Pass `columns` (the order the copula was fitted on) to get a labelled
    Series back, which is what `denormalize` needs in order to give each series
    its own uniform. Without it the raw positional vector is returned.
    """
    simulated = np.array(cop.simulate(1))
    sample = cast(np.ndarray[Any, np.dtype[np.float64]], simulated[0])
    if columns is None:
        return sample
    if len(columns) != len(sample):
        raise ValueError(
            f"Copula produced {len(sample)} uniforms but {len(columns)} column "
            "names were given; they must correspond one-to-one."
        )
    return pd.Series(sample, index=list(columns))


def uniforms_by_column(
    u_sample: np.ndarray | pd.Series | Mapping[str, float] | None,
    u_columns: Sequence[str] | None = None,
) -> dict[str, float] | None:
    """
    Normalizes the several accepted uniform-sample forms into a name -> uniform map.

    Positional ndarrays are only accepted alongside `u_columns`: mapping them by
    position is what silently scrambled the dependence structure, so a bare
    array is rejected rather than guessed at.
    """
    if u_sample is None:
        return None
    if isinstance(u_sample, pd.Series):
        return {str(k): float(v) for k, v in u_sample.items()}
    if isinstance(u_sample, Mapping):
        return {str(k): float(v) for k, v in u_sample.items()}

    values = np.asarray(u_sample, dtype=float).ravel()
    if u_columns is None:
        raise ValueError(
            "u_sample was given as an unlabelled array, so there is no way to "
            "tell which series each uniform belongs to. Pass a pd.Series "
            "indexed by series name (see VineFit.sample), or pass u_columns "
            "naming each entry in order."
        )
    if len(u_columns) != len(values):
        raise ValueError(
            f"u_sample has {len(values)} entries but u_columns names "
            f"{len(u_columns)}; they must correspond one-to-one."
        )
    return {str(c): float(v) for c, v in zip(u_columns, values)}
