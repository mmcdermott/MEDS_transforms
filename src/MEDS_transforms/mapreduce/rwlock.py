"""Locking functions."""

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from filelock import FileLock, Timeout
from omegaconf import DictConfig

from ..compute_modes import COMPUTE_FN_T
from ..dataframe import READ_FN_T, WRITE_FN_T

logger = logging.getLogger(__name__)

LOCK_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"
FILE_CHECKER_T = Callable[[Path], bool]

RUN_MARKER_DIRNAME = ".run_markers"


def is_complete_parquet_file(fp: Path) -> bool:
    """Check if a parquet file is complete.

    Args:
        fp: The file path to the parquet file.

    Returns:
        True if the parquet file is complete, False otherwise.

    Examples:
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> with tempfile.NamedTemporaryFile() as tmp:
        ...     df.write_parquet(tmp)
        ...     is_complete_parquet_file(tmp)
        True
        >>> with tempfile.NamedTemporaryFile() as tmp:
        ...     df.write_csv(tmp)
        ...     is_complete_parquet_file(tmp)
        False
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tmp = Path(tmp)
        ...     is_complete_parquet_file(tmp / "nonexistent.parquet")
        False
    """

    try:
        _ = pq.ParquetFile(fp)
        return True
    except Exception:
        return False


def default_file_checker(fp: Path) -> bool:
    """Check if a file exists and is complete."""
    if fp.suffix == ".parquet":
        return is_complete_parquet_file(fp)
    return fp.is_file()


def run_marker_dir(cfg: DictConfig) -> Path | None:
    """Resolve the directory in which this stage invocation records the outputs it has produced.

    Parallel map stages divide work by having every worker walk the same ``(input -> output)`` list and
    letting :func:`rwlock_wrap`'s "output already exists, skip it" branch arbitrate. ``do_overwrite=True``
    disables that branch, so workers stop sharing work and start deleting each other's fresh outputs. The
    cure is to let ``do_overwrite`` distinguish *"stale, from a previous run"* from *"fresh, produced by
    a sibling worker moments ago"*, which needs an identifier scoped to one invocation and shared by all
    of its workers.

    ``run_id`` is that identifier. It is stamped by the ``MEDS_transform-stage`` dispatcher before Hydra
    fans out, so every worker in a ``--multirun`` sweep inherits the same value, and every fresh
    invocation gets a new one. Markers live under ``log_dir`` (already a per-stage scratch area) so no
    bookkeeping files land next to the MEDS data.

    Args:
        cfg: The stage configuration. Uses ``run_id`` and ``log_dir``, both optional.

    Returns:
        The marker directory for this run, or ``None`` if the config carries no ``run_id`` or no
        ``log_dir`` — in which case ``do_overwrite`` keeps its historical unconditional-overwrite
        behavior.

    Examples:
        >>> cfg = DictConfig({"run_id": "abc123", "log_dir": "/data/out/.logs"})
        >>> run_marker_dir(cfg)
        PosixPath('/data/out/.logs/.run_markers/abc123')

        Absent either key, run scoping is off:

        >>> print(run_marker_dir(DictConfig({"log_dir": "/data/out/.logs"})))
        None
        >>> print(run_marker_dir(DictConfig({"run_id": None, "log_dir": "/data/out/.logs"})))
        None
        >>> print(run_marker_dir(DictConfig({"run_id": "abc123"})))
        None
    """

    run_id = cfg.get("run_id", None)
    log_dir = cfg.get("log_dir", None)

    if not run_id or not log_dir:
        return None

    return Path(log_dir) / RUN_MARKER_DIRNAME / str(run_id)


def _run_marker_fp(marker_dir: Path, out_fp: Path) -> Path:
    """Return the marker path recording that ``out_fp`` was produced during ``marker_dir``'s run.

    Output paths are hashed rather than mirrored into the marker directory: they are absolute, may live
    on a different tree entirely, and only ever need to be compared for equality.

    Examples:
        >>> marker_fp = _run_marker_fp(Path("/logs/.run_markers/abc"), Path("/data/train/0.parquet"))
        >>> marker_fp.parent
        PosixPath('/logs/.run_markers/abc')
        >>> marker_fp.suffix
        '.produced'

        The mapping is stable, and distinct outputs get distinct markers:

        >>> marker_fp == _run_marker_fp(Path("/logs/.run_markers/abc"), Path("/data/train/0.parquet"))
        True
        >>> marker_fp == _run_marker_fp(Path("/logs/.run_markers/abc"), Path("/data/train/1.parquet"))
        False
    """

    key = hashlib.sha256(str(Path(out_fp).resolve()).encode("utf-8")).hexdigest()
    return marker_dir / f"{key}.produced"


def rwlock_wrap(
    in_fp: Path,
    out_fp: Path,
    read_fn: READ_FN_T,
    write_fn: WRITE_FN_T,
    compute_fn: COMPUTE_FN_T,
    do_overwrite: bool = False,
    out_fp_checker: FILE_CHECKER_T = default_file_checker,
    marker_dir: Path | None = None,
) -> bool:
    """Wrap a series of file-in file-out map transformations on a dataframe with caching and locking.

    Args:
        in_fp: The file path of the input dataframe. Must exist and be readable via `read_fn`.
        out_fp: Output file path. The parent directory will be created if it does not exist. If this file
            already exists, it will be deleted before any computations are done if `do_overwrite=True`, which
            can result in data loss if the transformation functions do not complete successfully on
            intermediate steps. If `do_overwrite` is `False` and this file exists, the function will use the
            `read_fn` to read the file and return the dataframe directly.
        read_fn: Function that reads the dataframe from a file. This must take as input a Path object and
            return a dataframe. Ideally, this read function can make use of lazy
            loading to further accelerate unnecessary reads when resuming from intermediate cached steps.
        write_fn: Function that writes the dataframe to a file. This must take as input a dataframe and a Path
            object, and will write the dataframe to that file.
        compute_fn: A function that transform the dataframe, which must take as input and return a dataframe.
        do_overwrite: If True, the output file will be overwritten if it already exists. This is `False` by
            default.
        marker_dir: Directory in which to record the outputs produced by the current run, as resolved by
            `run_marker_dir`. When provided, `do_overwrite` is scoped to the run: outputs recorded here
            are treated as cache hits rather than overwritten, so parallel workers keep dividing work
            between themselves instead of redoing (and deleting) each other's fresh results. When `None`
            (the default), `do_overwrite=True` overwrites unconditionally, as it always has.

    Returns:
        True if the computation was run, False otherwise.

    Examples:
        >>> directory = tempfile.TemporaryDirectory()
        >>> read_fn = pl.read_csv
        >>> write_fn = pl.DataFrame.write_csv
        >>> root = Path(directory.name)
        >>> # For this example we'll use a simple CSV file, but in practice we *strongly* recommend using
        >>> # Parquet files for performance reasons.
        >>> in_fp = root / "input.csv"
        >>> out_fp = root / "output.csv"
        >>> in_df = pl.DataFrame({"a": [1, 3, 3], "b": [2, 4, 5], "c": [3, -1, 6]})
        >>> in_df.write_csv(in_fp)
        >>> def compute_fn(df: pl.DataFrame) -> pl.DataFrame:
        ...     return df.with_columns(pl.col("c") * 2).filter(pl.col("c") > 4)
        >>> result_computed = rwlock_wrap(in_fp, out_fp, read_fn, write_fn, compute_fn)
        >>> assert result_computed
        >>> print(out_fp.read_text())
        a,b,c
        1,2,6
        3,5,12
        <BLANKLINE>
        >>> in_df_2 = pl.DataFrame({"a": [1], "b": [3], "c": [-1]})
        >>> in_fp_2 = root / "input_2.csv"
        >>> in_df_2.write_csv(in_fp_2)
        >>> compute_fn = lambda df: df
        >>> result_computed = rwlock_wrap(in_fp_2, out_fp, read_fn, write_fn, compute_fn, do_overwrite=True)
        >>> assert result_computed
        >>> print(out_fp.read_text())
        a,b,c
        1,3,-1
        <BLANKLINE>
        >>> out_fp.unlink()
        >>> compute_fn = lambda df: df.with_columns(pl.col("c") * 2).filter(pl.col("d") > 4)
        >>> rwlock_wrap(in_fp, out_fp, read_fn, write_fn, compute_fn)
        Traceback (most recent call last):
            ...
        polars.exceptions.ColumnNotFoundError: unable to find column "d"; valid columns: ["a", "b", "c"]
        >>> assert not out_fp.is_file()  # Out file should not be created when the process crashes

    If the lock file already exists, the function will not do anything

        >>> def compute_fn(df: pl.DataFrame) -> pl.DataFrame:
        ...     return df.with_columns(pl.col("c") * 2).filter(pl.col("c") > 4)
        >>> out_fp = root / "output.csv"
        >>> lock_fp = root / "output.csv.lock"
        >>> with FileLock(str(lock_fp)):
        ...     result_computed = rwlock_wrap(in_fp, out_fp, read_fn, write_fn, compute_fn)
        ...     assert not result_computed

    The lock file will be removed after successful processing.

        >>> result_computed = rwlock_wrap(in_fp, out_fp, read_fn, write_fn, compute_fn)
        >>> assert result_computed
        >>> assert not lock_fp.exists()

    Passing a `marker_dir` scopes `do_overwrite` to a single run. The first visit to an output still
    overwrites whatever a previous run left there, but subsequent visits — which in a parallel map stage
    are other workers walking the same shard list — see the output as a cache hit and leave it alone:

        >>> n_computed = 0
        >>> def counting_compute_fn(df: pl.DataFrame) -> pl.DataFrame:
        ...     global n_computed
        ...     n_computed += 1
        ...     return df
        >>> out_fp = root / "run_scoped.csv"
        >>> _ = out_fp.write_text("stale output left over from an earlier run")
        >>> run_1 = root / "markers" / "run-1"
        >>> kwargs = {"do_overwrite": True, "marker_dir": run_1}
        >>> rwlock_wrap(in_fp, out_fp, read_fn, write_fn, counting_compute_fn, **kwargs)
        True
        >>> rwlock_wrap(in_fp, out_fp, read_fn, write_fn, counting_compute_fn, **kwargs)
        False
        >>> n_computed  # the stale output was replaced once, not once per worker
        1

    The output is never deleted on those repeat visits, which is what let a concurrent reducer read a
    path that had momentarily vanished:

        >>> print(out_fp.read_text())
        a,b,c
        1,2,3
        3,4,-1
        3,5,6
        <BLANKLINE>

    A later run passes a new `marker_dir` and so does overwrite, honoring `do_overwrite`:

        >>> rwlock_wrap(in_fp, out_fp, read_fn, write_fn, counting_compute_fn, do_overwrite=True,
        ...             marker_dir=root / "markers" / "run-2")
        True
        >>> n_computed
        2

    Without a `marker_dir`, `do_overwrite=True` overwrites on every visit, as it always has:

        >>> rwlock_wrap(in_fp, out_fp, read_fn, write_fn, counting_compute_fn, do_overwrite=True)
        True
        >>> rwlock_wrap(in_fp, out_fp, read_fn, write_fn, counting_compute_fn, do_overwrite=True)
        True
        >>> n_computed
        4
        >>> directory.cleanup()
    """

    # Markers only ever inform the `do_overwrite` branch, so without it there is nothing to record and
    # the default path touches the filesystem exactly as much as it did before.
    marker_fp = _run_marker_fp(marker_dir, out_fp) if (marker_dir is not None and do_overwrite) else None

    def is_reusable(out_exists: bool) -> bool:
        """Whether an existing `out_fp` can be reused as-is, without recomputing it."""
        if not out_exists:
            return False
        if not do_overwrite:
            return True
        # Under `do_overwrite`, only outputs this run produced are reusable; anything else is stale.
        return marker_fp is not None and marker_fp.is_file()

    # `out_fp_checker` reads the parquet footer, so its result is passed around rather than recomputed.
    if is_reusable(out_fp_checker(out_fp)):
        logger.info(f"{out_fp} exists; returning.")
        return False

    lock_fp = out_fp.with_suffix(f"{out_fp.suffix}.lock")
    lock = FileLock(str(lock_fp))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        logger.info(f"Lock found at {lock_fp}. Returning.")
        return False

    try:
        # Re-check under the lock: another worker may have finished this output between our check above
        # and our acquisition here, in which case recomputing it would be wasted work.
        out_exists = out_fp_checker(out_fp)
        if is_reusable(out_exists):
            logger.info(f"{out_fp} was produced while we waited for the lock; returning.")
            return False

        if out_exists:
            # Only reachable with `do_overwrite=True` on an output this run did not produce. Deleting
            # while holding the lock — rather than before acquiring it, as this used to — keeps the
            # output from being absent during a window in which nothing guards it.
            logger.info(f"Deleting existing {out_fp} as do_overwrite={do_overwrite}.")
            out_fp.unlink()

        st_time = datetime.now(tz=UTC)
        logger.info(f"Reading input dataframe from {in_fp}")
        df = read_fn(in_fp)
        logger.info("Read dataset")
        df = compute_fn(df)
        logger.info(f"Writing final output to {out_fp}")
        write_fn(df, out_fp)
        if marker_fp is not None:
            # Recorded before the lock is released, so no worker can observe a finished output that is
            # not yet marked and conclude it is stale.
            marker_fp.parent.mkdir(parents=True, exist_ok=True)
            marker_fp.write_text(str(out_fp))
        logger.info(f"Succeeded in {datetime.now(tz=UTC) - st_time}")
        return True
    finally:
        lock.release()
        lock_fp.unlink(missing_ok=True)
