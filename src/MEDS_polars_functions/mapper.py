"""Basic utilities for parallelizable map operations on sharded MEDS datasets with caching and locking."""

import json
import random
import shutil
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe

LOCK_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def get_earliest_lock(cache_directory: Path) -> datetime | None:
    """Returns the earliest start time of any lock file present in a cache directory, or None if none exist.

    Args:
        cache_directory: The cache directory to check for the presence of a lock file.

    Examples:
        >>> import tempfile
        >>> directory = tempfile.TemporaryDirectory()
        >>> root = Path(directory.name)
        >>> empty_directory = root / "cache_empty"
        >>> empty_directory.mkdir(exist_ok=True, parents=True)
        >>> cache_directory = root / "cache_with_locks"
        >>> locks_directory = cache_directory / "locks"
        >>> locks_directory.mkdir(exist_ok=True, parents=True)
        >>> time_1 = datetime(2021, 1, 1)
        >>> time_1_str = time_1.strftime(LOCK_TIME_FMT) # "2021-01-01T00:00:00.000000"
        >>> lock_fp_1 = locks_directory / f"{time_1_str}.json"
        >>> _ = lock_fp_1.write_text(json.dumps({"start": time_1_str}))
        >>> time_2 = datetime(2021, 1, 2, 3, 4, 5)
        >>> time_2_str = time_2.strftime(LOCK_TIME_FMT) # "2021-01-02T03:04:05.000000"
        >>> lock_fp_2 = locks_directory / f"{time_2_str}.json"
        >>> _ = lock_fp_2.write_text(json.dumps({"start": time_2_str}))
        >>> get_earliest_lock(cache_directory)
        datetime.datetime(2021, 1, 1, 0, 0)
        >>> get_earliest_lock(empty_directory) is None
        True
        >>> lock_fp_1.unlink()
        >>> get_earliest_lock(cache_directory)
        datetime.datetime(2021, 1, 2, 3, 4, 5)
        >>> directory.cleanup()
    """
    locks_directory = cache_directory / "locks"

    lock_times = [
        datetime.strptime(json.loads(lock_fp.read_text())["start"], LOCK_TIME_FMT)
        for lock_fp in locks_directory.glob("*.json")
    ]

    return min(lock_times) if lock_times else None


def register_lock(cache_directory: Path) -> tuple[datetime, Path]:
    """Register a lock file in a cache directory.

    Args:
        cache_directory: The cache directory to register a lock file in.

    Examples:
        >>> import tempfile
        >>> directory = tempfile.TemporaryDirectory()
        >>> root = Path(directory.name)
        >>> cache_directory = root / "cache_with_locks"
        >>> lock_time, lock_fp = register_lock(cache_directory)
        >>> assert (datetime.now() - lock_time).total_seconds() < 1, "Lock time should be ~ now."
        >>> lock_fp.is_file()
        True
        >>> lock_fp.read_text() == f'{{"start": "{lock_time.strftime(LOCK_TIME_FMT)}"}}'
        True
        >>> directory.cleanup()
    """

    lock_directory = cache_directory / "locks"
    lock_directory.mkdir(exist_ok=True, parents=True)

    lock_time = datetime.now()
    lock_fp = lock_directory / f"{lock_time.strftime(LOCK_TIME_FMT)}.json"
    lock_fp.write_text(json.dumps({"start": lock_time.strftime(LOCK_TIME_FMT)}))
    return lock_time, lock_fp


def rwlock_wrap[
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
    do_return: bool = False,
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
        do_return: If True, the final dataframe will be returned. This is `False` by default.

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
        >>> result_computed = rwlock_wrap(in_fp, out_fp, read_fn, write_fn, *transform_fns, do_return=False)
        >>> assert result_computed
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
        >>> rwlock_wrap(in_fp, out_fp, read_fn, write_fn, *transform_fns)
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
        >>> lock_dir = cache_directory / "locks"
        >>> assert not lock_dir.exists()
        >>> def lock_dir_checker_fn(df: pl.DataFrame) -> pl.DataFrame:
        ...     print(f"Lock dir exists? {lock_dir.exists()}")
        ...     return df
        >>> result_computed, out_df = rwlock_wrap(
        ...     in_fp, out_fp, read_fn, write_fn, lock_dir_checker_fn, do_return=True
        ... )
        Lock dir exists? True
        >>> assert result_computed
        >>> out_df
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
            if do_return:
                return True, read_fn(out_fp)
            else:
                return True

    cache_directory = out_fp.parent / f".{out_fp.stem}_cache"
    cache_directory.mkdir(exist_ok=True, parents=True)

    earliest_lock_time = get_earliest_lock(cache_directory)
    if earliest_lock_time is not None:
        logger.info(f"{out_fp} is in progress as of {earliest_lock_time}. Returning.")
        return False, None if do_return else False

    st_time, lock_fp = register_lock(cache_directory)

    logger.info(f"Registered lock at {st_time}. Double checking no earlier locks have been registered.")
    earliest_lock_time = get_earliest_lock(cache_directory)
    if earliest_lock_time < st_time:
        logger.info(f"Earlier lock found at {earliest_lock_time}. Deleting current lock and returning.")
        lock_fp.unlink()
        return False, None if do_return else False

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

            if cache_intermediate and i < len(transform_fns) - 1:
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
        if do_return:
            return True, df
        else:
            return True
    except Exception as e:
        logger.warning(f"Clearing lock due to Exception {e} at {lock_fp} after {datetime.now() - st_time}")
        lock_fp.unlink()
        raise e


def get_paths_and_debug(cfg: DictConfig, stage_cfg: DictConfig):
    input_dir = Path(stage_cfg.data_input_dir)
    output_dir = Path(stage_cfg.output_dir)
    metadata_input_dir = Path(stage_cfg.metadata_input_dir)
    shards_map_fn = Path(stage_cfg.shards_map_fp)

    def chk(x: Path):
        return "✅" if x.exists() else "❌"

    paths_strs = [
        f"  - {k}: {chk(v)} {str(v.resolve())}"
        for k, v in {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "metadata_input_dir": metadata_input_dir,
            "shards_map_fn": shards_map_fn,
        }.items()
    ]

    logger_strs = [
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}",
        f"Stage: {cfg.stage}",
        f"Stage config:\n{OmegaConf.to_yaml(stage_cfg)}",
        "Paths: (checkbox indicates if it exists)",
    ]
    logger.debug("\n".join(logger_strs + paths_strs))

    return input_dir, output_dir, metadata_input_dir, shards_map_fn


def map_over[
    DF_T
](
    config_path: str,
    config_name: str,
    read_fn: Callable[[Path], DF_T] = pl.scan_parquet,
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    version_base=None,
):
    """A wrapper for doing a mapreduce over a set of shards with a mapper function."""
    hydra_kwargs = {"version_base": version_base, "config_path": config_path, "config_name": config_name}

    def decorator(mapper_fn: callable):
        @wraps(mapper_fn)
        def wrapper(cfg: DictConfig):
            hydra_loguru_init()

            stage_cfg = cfg.stage_cfg

            input_dir, output_dir, metadata_input_dir, shards_map_fn = get_paths_and_debug(cfg, stage_cfg)

            shards = json.loads(shards_map_fn.read_text())

            if "process_shard_prefix" in stage_cfg:
                logger.info(f'Processing shards with prefix "{stage_cfg.process_shard_prefix}"')
                shards = {k: v for k, v in shards.items() if k.startswith(stage_cfg.process_shard_prefix)}

            shards = list(shards.keys())
            random.shuffle(shards)

            start = datetime.now()
            logger.info(f"Mapping computation over {len(shards)} shards")

            for sp in shards:
                in_fp = input_dir / f"{sp}.parquet"
                out_fp = output_dir / f"{sp}.parquet"

                logger.info(f"Processing {str(in_fp.resolve())} into {str(out_fp.resolve())}")

                rwlock_wrap(
                    in_fp,
                    out_fp,
                    read_fn,
                    write_fn,
                    mapper_fn,
                    do_return=False,
                    cache_intermediate=False,
                    do_overwrite=cfg.do_overwrite,
                )

            logger.info(f"Finished mapping in {datetime.now() - start}")

        return hydra.main(**hydra_kwargs)(wrapper)

    return decorator
