"""Basic utilities for parallelizable mapreduces on sharded MEDS datasets with caching and locking."""

import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

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
    compute_fn: Callable[[DF_T], DF_T],
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
        compute_fn: A function that transform the dataframe, which must take as input a dataframe of (generic)
            type DF_T and return a dataframe of (generic) type DF_T.
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
        >>> compute_fn = lambda df: df.with_columns(pl.col("c") * 2).filter(pl.col("c") > 4)
        >>> result_computed = rwlock_wrap(in_fp, out_fp, read_fn, write_fn, compute_fn, do_return=False)
        >>> assert result_computed
        >>> print(out_fp.read_text())
        a,b,c
        1,2,6
        3,5,12
        <BLANKLINE>
        >>> out_fp.unlink()
        >>> compute_fn = lambda df: df.with_columns(pl.col("c") * 2).filter(pl.col("d") > 4)
        >>> rwlock_wrap(in_fp, out_fp, read_fn, write_fn, compute_fn)
        Traceback (most recent call last):
            ...
        polars.exceptions.ColumnNotFoundError: unable to find column "d"; valid columns: ["a", "b", "c"]
        >>> cache_directory = root / f".output_cache"
        >>> lock_dir = cache_directory / "locks"
        >>> assert not list(lock_dir.iterdir())
        >>> def lock_dir_checker_fn(df: pl.DataFrame) -> pl.DataFrame:
        ...     print(f"Lock dir empty? {not (list(lock_dir.iterdir()))}")
        ...     return df
        >>> result_computed, out_df = rwlock_wrap(
        ...     in_fp, out_fp, read_fn, write_fn, lock_dir_checker_fn, do_return=True
        ... )
        Lock dir empty? False
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
        df = compute_fn(df)
        logger.info(f"Writing final output to {out_fp}")
        write_fn(df, out_fp)
        logger.info(f"Succeeded in {datetime.now() - st_time}")
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


def shard_iterator(
    cfg: DictConfig,
    in_suffix: str = ".parquet",
    out_suffix: str = ".parquet",
    in_prefix: str = "",
    out_prefix: str = "",
):
    """Provides a generator that yields shard input and output files for mapreduce operations.

    Args:
        cfg: The configuration dictionary for the overall pipeline. Should (possibly) contain the following
            keys (some are optional, as marked below):
            - ``stage_cfg.data_input_dir`` (mandatory): The directory containing the input data.
            - ``stage_cfg.output_dir`` (mandatory): The directory to write the output data.
            - ``shards_map_fp`` (mandatory): The file path to the shards map JSON file.
            - ``stage_cfg.process_split`` (optional): The prefix of the shards to process (e.g.,
              ``"train/"``). If not provided, all shards will be processed.
            - ``worker`` (optional): The worker ID for the MR worker; this is also used to seed the
              randomization process. If not provided, the randomization process is unseeded.
        in_suffix: The suffix of the input files. Defaults to ".parquet". This can be set to "" to process
            entire directories.
        out_suffix: The suffix of the output files. Defaults to ".parquet".
        in_prefix: The prefix of the input files. Defaults to "". This can be used to load files from a
            subdirectory of the input directory by including a "/" at the end of the prefix.
        out_prefix: The prefix of the output files. Defaults to "".

    Yields:
        Randomly shuffled pairs of input and output file paths for each shard. The randomization process is
        seeded by the worker ID in ``cfg``, if provided, otherwise it is left unseeded.

    Examples:
        >>> from tempfile import NamedTemporaryFile
        >>> shards = {"train/0": [1, 2, 3], "train/1": [4, 5, 6], "held_out": [4, 5, 6], "foo": [5]}
        >>> with NamedTemporaryFile() as tmp:
        ...     _ = Path(tmp.name).write_text(json.dumps(shards))
        ...     cfg = DictConfig({
        ...         "stage_cfg": {"data_input_dir": "data/", "output_dir": "output/"},
        ...         "shards_map_fp": tmp.name, "worker": 1,
        ...     })
        ...     gen = shard_iterator(cfg)
        ...     list(gen) # doctest: +NORMALIZE_WHITESPACE
        [(PosixPath('data/foo.parquet'),      PosixPath('output/foo.parquet')),
         (PosixPath('data/train/0.parquet'),  PosixPath('output/train/0.parquet')),
         (PosixPath('data/held_out.parquet'), PosixPath('output/held_out.parquet')),
         (PosixPath('data/train/1.parquet'),  PosixPath('output/train/1.parquet'))]
        >>> with NamedTemporaryFile() as tmp:
        ...     _ = Path(tmp.name).write_text(json.dumps(shards))
        ...     cfg = DictConfig({
        ...         "stage_cfg": {"data_input_dir": "data/", "output_dir": "output/"},
        ...         "shards_map_fp": tmp.name, "worker": 1,
        ...     })
        ...     gen = shard_iterator(cfg, in_suffix="", out_suffix=".csv", in_prefix="a/", out_prefix="b/")
        ...     list(gen) # doctest: +NORMALIZE_WHITESPACE
        [(PosixPath('data/a/foo'),      PosixPath('output/b/foo.csv')),
         (PosixPath('data/a/train/0'),  PosixPath('output/b/train/0.csv')),
         (PosixPath('data/a/held_out'), PosixPath('output/b/held_out.csv')),
         (PosixPath('data/a/train/1'),  PosixPath('output/b/train/1.csv'))]
        >>> with NamedTemporaryFile() as tmp:
        ...     _ = Path(tmp.name).write_text(json.dumps(shards))
        ...     cfg = DictConfig({
        ...         "stage_cfg": {
        ...             "data_input_dir": "data/", "output_dir": "output/", "process_split": "train/"
        ...         },
        ...         "shards_map_fp": tmp.name,
        ...         "worker": 1,
        ...     })
        ...     gen = shard_iterator(cfg)
        ...     list(gen) # doctest: +NORMALIZE_WHITESPACE
        [(PosixPath('data/train/1.parquet'),  PosixPath('output/train/1.parquet')),
         (PosixPath('data/train/0.parquet'),  PosixPath('output/train/0.parquet'))]

    Note that if you ask it to process a split that isn't a valid prefix of the shards, it will return all
    shards and assume that is covered via a `patient_splits.parquet` file:
        >>> with NamedTemporaryFile() as tmp:
        ...     _ = Path(tmp.name).write_text(json.dumps(shards))
        ...     cfg = DictConfig({
        ...         "stage_cfg": {"data_input_dir": "data/", "output_dir": "output/"},
        ...         "shards_map_fp": tmp.name, "process_split": "nonexisting", "worker": 1,
        ...     })
        ...     gen = shard_iterator(cfg)
        ...     list(gen) # doctest: +NORMALIZE_WHITESPACE
        [(PosixPath('data/foo.parquet'),      PosixPath('output/foo.parquet')),
         (PosixPath('data/train/0.parquet'),  PosixPath('output/train/0.parquet')),
         (PosixPath('data/held_out.parquet'), PosixPath('output/held_out.parquet')),
         (PosixPath('data/train/1.parquet'),  PosixPath('output/train/1.parquet'))]
    """

    input_dir = Path(cfg.stage_cfg.data_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)
    shards_map_fp = Path(cfg.shards_map_fp) if "shards_map_fp" in cfg else None

    if shards_map_fp and shards_map_fp.is_file():
        shards = json.loads(shards_map_fp.read_text())

        if "process_split" in cfg.stage_cfg:
            if any(k.startswith(cfg.stage_cfg.process_split) for k in shards):
                logger.info(f'Processing shards with prefix "{cfg.stage_cfg.process_split}"')
                shards = {k: v for k, v in shards.items() if k.startswith(cfg.stage_cfg.process_split)}
        shards = list(shards.keys())
    else:
        shards = []
        for p in input_dir.glob(f"{in_prefix}*{in_suffix}"):
            relative_path = p.relative_to(input_dir)
            shard_name = str(relative_path)
            shard_name = shard_name[len(in_prefix) :] if in_prefix else shard_name
            shard_name = shard_name[: -len(in_suffix)] if in_suffix else shard_name
            shards.append(shard_name)

    if "worker" in cfg:
        random.seed(cfg.worker)
    random.shuffle(shards)

    logger.info(f"Mapping computation over a maximum of {len(shards)} shards")

    for sp in shards:
        in_fp = input_dir / f"{in_prefix}{sp}{in_suffix}"
        out_fp = output_dir / f"{out_prefix}{sp}{out_suffix}"

        # TODO: Could add checking logic for existence of in_fp and/or out_fp here.

        yield in_fp, out_fp
