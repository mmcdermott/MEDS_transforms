"""Locking functions."""

from collections.abc import Callable
from datetime import datetime, UTC
import logging
from pathlib import Path

from filelock import FileLock, Timeout
import pyarrow.parquet as pq

from ..compute_modes import COMPUTE_FN_T
from ..dataframe import READ_FN_T, WRITE_FN_T

logger = logging.getLogger(__name__)

LOCK_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"
FILE_CHECKER_T = Callable[[Path], bool]


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


def rwlock_wrap(
    in_fp: Path,
    out_fp: Path,
    read_fn: READ_FN_T,
    write_fn: WRITE_FN_T,
    compute_fn: COMPUTE_FN_T,
    do_overwrite: bool = False,
    out_fp_checker: FILE_CHECKER_T = default_file_checker,
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
        ...
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
    """

    if out_fp_checker(out_fp):
        if do_overwrite:
            logger.info(f"Deleting existing {out_fp} as do_overwrite={do_overwrite}.")
            out_fp.unlink()
        else:
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
        st_time = datetime.now(tz=UTC)
        logger.info(f"Reading input dataframe from {in_fp}")
        df = read_fn(in_fp)
        logger.info("Read dataset")
        df = compute_fn(df)
        logger.info(f"Writing final output to {out_fp}")
        write_fn(df, out_fp)
        logger.info(f"Succeeded in {datetime.now(tz=UTC) - st_time}")
        return True
    finally:
        lock.release()
        try:
            lock_fp.unlink()
        except FileNotFoundError:  # pragma: no cover
            logger.warning(f"Lock file {lock_fp} was not found, though was successfully released. Returning")
