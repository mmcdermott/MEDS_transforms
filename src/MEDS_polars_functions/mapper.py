import json
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from loguru import logger


def wrap[
    DF_T
](
    in_fp: Path,
    out_fp: Path,
    read_fn: Callable[[Path], DF_T],
    write_fn: Callable[[DF_T, Path], None],
    *transform_fns: Callable[[DF_T], DF_T],
    cache_intermediate: bool = True,
    clear_cache_on_completion: bool = True,
    do_overwrite: bool = False,
) -> tuple[bool, DF_T | None]:
    """Wrap a series of file-in file-out map transformations on a dataframe with caching and locking.

    Args:
        in_fp: The file path of the input dataframe. Must exist and be readable via `read_fn`.
        out_fp: Output file path. The parent directory will be created if it does not exist. If this file
            already exists, it will be deleted before any computations are done if `do_overwrite=True`, which
            can result in data loss if the transformation functions do not complete successfully on
            intermediate steps. If `do_overwrite` is `False` and this file exists, the function will use the
            `read_fn` to read the file and return the dataframe directly.
        read_fn: Function that reads the dataframe from a file. This must take as input a Path object and
            return a dataframe of (generic) type DF_T. Ideally, this read function can make use of lazy
            loading to further accelerate unnecessary reads when resuming from intermediate cached steps.
        write_fn: Function that writes the dataframe to a file. This must take as input a dataframe of
            (generic) type DF_T and a Path object, and will write the dataframe to that file.
        transform_fns: A series of functions that transform the dataframe. Each function must take as input
            a dataframe of (generic) type DF_T and return a dataframe of (generic) type DF_T. The functions
            will be applied in the passed order.
        cache_intermediate: If True, intermediate outputs of the transformations will be cached in a hidden
            directory in the same parent directory as `out_fp` of the form
            `{out_fp.parent}/.{out_fp.stem}_cache`. This can be useful for debugging and resuming from
            intermediate steps when nontrivial transformations are composed. Cached files will be named
            `step_{i}.output` where `i` is the index of the transformation function in `transform_fns`. **Note
            that if you change the order of the transformations, the cache will be no longer valid but the
            system will _not_ automatically delete the cache!**. This is `True` by default.
            If `do_overwrite=True`, any prior individual cache files that are detected during the run will be
            deleted before their corresponding step is run. If `do_overwrite=False` and a cache file exists,
            that step of the transformation will be skipped and the cache file will be read directly.
        clear_cache_on_completion: If True, the cache directory will be deleted after the final output is
            written. This is `True` by default.
        do_overwrite: If True, the output file will be overwritten if it already exists. This is `False` by
            default.

    Returns:
        The dataframe resulting from the transformations applied in sequence to the dataframe stored in
        `in_fp`.

    Examples:
        >>> import polars as pl
        >>> import tempfile
        >>> directory = tempfile.TemporaryDirectory()
        >>> root = Path(directory.name)
        >>> # For this example we'll use a simple CSV file, but in practice we *strongly* recommend using
        >>> # Parquet files for performance reasons.
        >>> in_fp = root / "input.csv"
        >>> out_fp = root / "output.csv"
        >>> in_df = pl.DataFrame({"a": [1, 3, 3], "b": [2, 4, 5], "c": [3, -1, 6]})
        >>> in_df.write_csv(in_fp)
        >>> read_fn = pl.read_csv
        >>> write_fn = pl.DataFrame.write_csv
        >>> transform_fns = [
        ...     lambda df: df.with_columns(pl.col("c") * 2),
        ...     lambda df: df.filter(pl.col("c") > 4)
        ... ]
        >>> wrap(in_fp, out_fp, read_fn, write_fn, *transform_fns)
        shape: (2, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 6   │
        │ 3   ┆ 5   ┆ 12  │
        └─────┴─────┴─────┘
        >>> print(out_fp.read_text())
        a,b,c
        1,2,6
        3,5,12
        <BLANKLINE>
        >>> out_fp.unlink()
        >>> cache_directory = root / f".output_cache"
        >>> assert not cache_directory.is_dir()
        >>> transform_fns = [
        ...     lambda df: df.with_columns(pl.col("c") * 2),
        ...     lambda df: df.filter(pl.col("d") > 4)
        ... ]
        >>> wrap(in_fp, out_fp, read_fn, write_fn, *transform_fns)
        Traceback (most recent call last):
            ...
        polars.exceptions.ColumnNotFoundError: unable to find column "d"; valid columns: ["a", "b", "c"]
        >>> assert cache_directory.is_dir()
        >>> cache_fp = cache_directory / "step_0.output"
        >>> pl.read_csv(cache_fp)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 6   │
        │ 3   ┆ 4   ┆ -2  │
        │ 3   ┆ 5   ┆ 12  │
        └─────┴─────┴─────┘
        >>> shutil.rmtree(cache_directory)
        >>> lock_fp = cache_directory / "lock.json"
        >>> assert not lock_fp.is_file()
        >>> def lock_fp_checker_fn(df: pl.DataFrame) -> pl.DataFrame:
        ...     print(f"Lock fp exists? {lock_fp.is_file()}")
        ...     return df
        >>> wrap(in_fp, out_fp, read_fn, write_fn, lock_fp_checker_fn)
        Lock fp exists? True
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ a   ┆ b   ┆ c   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╪═════╡
        │ 1   ┆ 2   ┆ 3   │
        │ 3   ┆ 4   ┆ -1  │
        │ 3   ┆ 5   ┆ 6   │
        └─────┴─────┴─────┘
        >>> directory.cleanup()
    """

    if out_fp.is_file():
        if do_overwrite:
            logger.info(f"Deleting existing {out_fp} as do_overwrite={do_overwrite}.")
            out_fp.unlink()
        else:
            logger.info(f"{out_fp} exists; reading directly and returning.")
            return True, read_fn(out_fp)

    cache_directory = out_fp.parent / f".{out_fp.stem}_cache"
    cache_directory.mkdir(exist_ok=True, parents=True)

    st_time = datetime.now()
    runtime_info = {"start": str(st_time)}

    lock_fp = cache_directory / "lock.json"
    if lock_fp.is_file():
        started_at = json.loads(lock_fp.read_text())["start"]
        logger.info(
            f"{out_fp} is under construction as of {started_at} as {lock_fp} exists. " "Returning None."
        )
        return False, None

    lock_fp.write_text(json.dumps(runtime_info))

    logger.info(f"Reading input dataframe from {in_fp}")
    df = read_fn(in_fp)
    logger.info("Read dataset")

    try:
        for i, transform_fn in enumerate(transform_fns):
            cache_fp = cache_directory / f"step_{i}.output"

            st_time_step = datetime.now()
            if cache_fp.is_file():
                if do_overwrite:
                    logger.info(
                        f"Deleting existing cached output for step {i} " f"as do_overwrite={do_overwrite}"
                    )
                    cache_fp.unlink()
                else:
                    logger.info(f"Reading cached output for step {i}")
                    df = read_fn(cache_fp)
            else:
                df = transform_fn(df)

            if cache_intermediate and i < len(transform_fns):
                logger.info(f"Writing intermediate output for step {i} to {cache_fp}")
                write_fn(df, cache_fp)
            logger.info(f"Completed step {i} in {datetime.now() - st_time_step}")

        logger.info(f"Writing final output to {out_fp}")
        write_fn(df, out_fp)
        logger.info(f"Succeeded in {datetime.now() - st_time}")
        if clear_cache_on_completion:
            logger.info(f"Clearing cache directory {cache_directory}")
            shutil.rmtree(cache_directory)
        else:
            logger.info(f"Leaving cache directory {cache_directory}, but clearing lock at {lock_fp}")
            lock_fp.unlink()
        return df
    except Exception as e:
        logger.warning(f"Clearing lock due to Exception {e} at {lock_fp} after {datetime.now() - st_time}")
        lock_fp.unlink()
        raise e
