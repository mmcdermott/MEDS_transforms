"""Basic utilities for serialized reduce operations on sharded MEDS datasets with caching and locking."""

import logging
import time
from pathlib import Path
from typing import Protocol

import polars as pl

from ..dataframe import DF_T, READ_FN_T, WRITE_FN_T
from .rwlock import default_file_checker

logger = logging.getLogger(__name__)


class REDUCE_FN_T(Protocol):  # noqa: N801
    """Protocol for a function that takes a variable number dataframes and returns one dataframe."""

    def __call__(self, *dfs: DF_T) -> DF_T: ...


def reduce_over(
    in_fps: list[Path],
    out_fp: Path,
    read_fn: READ_FN_T,
    write_fn: WRITE_FN_T,
    reduce_fn: REDUCE_FN_T,
    merge_fp: Path | None = None,
    merge_fn: REDUCE_FN_T | None = None,
    do_overwrite: bool = False,
    polling_time: float = 0.1,
    max_poll_time: float | None = None,
):
    """Performs a reduction operation on a list of input file paths, with optional merging to existing data.

    Args:
        in_fps: List of input file paths containing data over which the reduction should be performed.
        out_fp: Output file path where the reduced data will be saved.
        polling_time: Time in seconds to wait between checks for file readiness.
        max_poll_time: Optional maximum total seconds to wait for input files to become readable.
            Defaults to ``None`` (wait indefinitely). When set, ``TimeoutError`` is raised if any
            input file exists but stays invalid past the deadline — guards against hangs when a
            mapper is permanently stuck or produced a corrupt file. Callers should pick a value
            meaningfully larger than ``polling_time``.
        read_fn: Function to read data from the input file paths.
        write_fn: Function to write the reduced data to the output file path.
        reduce_fn: Function to perform the reduction operation on the data. It should take two dataframe
            arguments and return one dataframe
        merge_fp: If this file exists, merge the output with the data stored in this file before finalization.
        merge_fn: A special reducer to perform the merging, if `merge_fp` is specified and is a file.
        do_overwrite: Should this overwrite an existing out file?

    Raises:
        FileExistsError: If the output file already exists.
        TimeoutError: If input files remain unreadable after ``max_poll_time`` seconds (only when set).

    Examples:
        >>> def reduce_fn(*dfs: pl.DataFrame) -> pl.DataFrame:
        ...     return pl.concat(dfs, how="vertical")
        >>> kwargs = {
        ...     "read_fn": pl.read_parquet,
        ...     "write_fn": pl.DataFrame.write_parquet,
        ...     "reduce_fn": reduce_fn,
        ... }
        >>> dfs = [pl.DataFrame({"a": [i, i + 1], "b": [i + 2, i + 3]}) for i in range(3)]
        >>> def write_dfs(in_fps: list[Path], dfs: list[pl.DataFrame] = dfs, delay_per: float = 0):
        ...     for i, (fp, df) in enumerate(zip(in_fps, dfs)):
        ...         time.sleep(delay_per)  # simulate slow write
        ...         df.write_parquet(fp)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     in_fps = [Path(tmpdir) / f"input_{i}.parquet" for i in range(3)]
        ...     write_dfs(in_fps)
        ...     out_fp = Path(tmpdir) / "output.parquet"
        ...     reduce_over(in_fps, out_fp, **kwargs)
        ...     pl.read_parquet(out_fp)
        shape: (6, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i8  ┆ i8  │
        ╞═════╪═════╡
        │ 0   ┆ 2   │
        │ 1   ┆ 3   │
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        │ 2   ┆ 4   │
        │ 3   ┆ 5   │
        └─────┴─────┘

    If specified, the reducer will merge with the `merge_fp` via the specialized function

        >>> merge_df = pl.DataFrame({"a": [-1, -2], "b": [-3, -4]})
        >>> def merge_fn(new: pl.DataFrame, old: pl.DataFrame) -> pl.DataFrame:
        ...     return pl.concat([old, new], how="vertical_relaxed")
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     in_fps = [Path(tmpdir) / f"input_{i}.parquet" for i in range(3)]
        ...     write_dfs(in_fps)
        ...     merge_fp = Path(tmpdir) / "merge.parquet"
        ...     merge_df.write_parquet(merge_fp)
        ...     out_fp = Path(tmpdir) / "output.parquet"
        ...     reduce_over(in_fps, out_fp, merge_fp=merge_fp, merge_fn=merge_fn, **kwargs)
        ...     pl.read_parquet(out_fp)
        shape: (8, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i8  ┆ i8  │
        ╞═════╪═════╡
        │ -1  ┆ -3  │
        │ -2  ┆ -4  │
        │ 0   ┆ 2   │
        │ 1   ┆ 3   │
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        │ 2   ┆ 4   │
        │ 3   ┆ 5   │
        └─────┴─────┘


    The reducer will wait for all the input files to exist before performing the computation:

        >>> import threading
        >>> def profile_reduce(dfs: list[pl.DataFrame], in_fps: list[Path], delay_per: float, out_fp: Path):
        ...     print("------------------")
        ...     print(f"Writing files with a delay of {delay_per} seconds...")
        ...     thread = threading.Thread(target=write_dfs, args=(in_fps, dfs, delay_per))
        ...     thread.daemon = True
        ...     st = datetime.now()
        ...     thread.start()
        ...     print("Starting reduction...")
        ...     reduce_over(in_fps, out_fp, **kwargs)
        ...     print(f"Reduction completed in: ~{(datetime.now() - st).total_seconds():.1f} seconds")
        ...     print(pl.read_parquet(out_fp))
        ...     thread.join()
        ...     for fp in in_fps:
        ...         fp.unlink()
        ...     out_fp.unlink()
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     in_fps = [Path(tmpdir) / f"input_{i}.parquet" for i in range(3)]
        ...     out_fp = Path(tmpdir) / "output.parquet"
        ...     profile_reduce(dfs, in_fps, 0.1, out_fp)
        ...     profile_reduce(dfs, in_fps, 0.5, out_fp)
        ------------------
        Writing files with a delay of 0.1 seconds...
        Starting reduction...
        Reduction completed in: ~0... seconds
        shape: (6, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i8  ┆ i8  │
        ╞═════╪═════╡
        │ 0   ┆ 2   │
        │ 1   ┆ 3   │
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        │ 2   ┆ 4   │
        │ 3   ┆ 5   │
        └─────┴─────┘
        ------------------
        Writing files with a delay of 0.5 seconds...
        Starting reduction...
        Reduction completed in: ~1... seconds
        shape: (6, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i8  ┆ i8  │
        ╞═════╪═════╡
        │ 0   ┆ 2   │
        │ 1   ┆ 3   │
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        │ 2   ┆ 4   │
        │ 3   ┆ 5   │
        └─────┴─────┘

    The reducer will error if the output file already exists, unless `do_overwrite` is set to `True`:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     in_fps = [Path(tmpdir) / f"input_{i}.parquet" for i in range(3)]
        ...     write_dfs(in_fps)
        ...     out_fp = Path(tmpdir) / "output.parquet"
        ...     out_fp.touch()
        ...     reduce_over(in_fps, out_fp, **kwargs)
        Traceback (most recent call last):
            ...
        FileExistsError: Output file already exists: /tmp/tmp.../output.parquet
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     in_fps = [Path(tmpdir) / f"input_{i}.parquet" for i in range(3)]
        ...     write_dfs(in_fps)
        ...     out_fp = Path(tmpdir) / "output.parquet"
        ...     out_fp.touch()
        ...     reduce_over(in_fps, out_fp, do_overwrite=True, **kwargs)
        ...     pl.read_parquet(out_fp)
        shape: (6, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i8  ┆ i8  │
        ╞═════╪═════╡
        │ 0   ┆ 2   │
        │ 1   ┆ 3   │
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        │ 2   ┆ 4   │
        │ 3   ┆ 5   │
        └─────┴─────┘
    """

    if out_fp.is_file() and not do_overwrite:
        raise FileExistsError(f"Output file already exists: {out_fp.resolve()!s}")

    if max_poll_time is not None and max_poll_time <= polling_time:
        raise ValueError(
            f"max_poll_time ({max_poll_time}s) must be greater than polling_time ({polling_time}s); "
            "otherwise legitimately slow mappers will trigger a TimeoutError on the first recheck."
        )

    deadline = None if max_poll_time is None else time.monotonic() + max_poll_time
    ready: set[Path] = set()

    def _is_ready(fp: Path) -> bool:
        # ``is_file`` is a cheap stat; only if it passes do we open the parquet to check completeness.
        return fp.is_file() and default_file_checker(fp)

    while True:
        ready.update(fp for fp in in_fps if fp not in ready and _is_ready(fp))
        if len(ready) == len(in_fps):
            break

        if deadline is None:
            sleep_for = polling_time
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pending = [fp for fp in in_fps if fp not in ready]
                stuck = [fp for fp in pending if fp.exists()]
                missing = [fp for fp in pending if not fp.exists()]
                parts = []
                if stuck:
                    parts.append(f"present but unreadable: {', '.join(str(fp) for fp in stuck)}")
                if missing:
                    parts.append(f"missing: {', '.join(str(fp) for fp in missing)}")
                raise TimeoutError(
                    f"Timed out after {max_poll_time}s waiting for reduction inputs — " + "; ".join(parts)
                )
            # Don't oversleep past the deadline; if remaining < polling_time, wake up sooner.
            sleep_for = min(polling_time, remaining)

        logger.info("Waiting to begin reduction for all files to be written...")
        time.sleep(sleep_for)

    reduced = reduce_fn(*[read_fn(fp) for fp in in_fps])

    if merge_fp is not None and merge_fp.is_file():
        reduced = merge_fn(reduced, read_fn(merge_fp))

    if isinstance(reduced, pl.LazyFrame):
        reduced = reduced.collect()
    reduced = reduced.with_columns(s.shrink_dtype() for s in reduced if s.dtype.is_numeric())

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    write_fn(reduced, out_fp)
