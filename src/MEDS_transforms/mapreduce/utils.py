"""Basic utilities for parallelizable mapreduces on sharded MEDS datasets with caching and locking."""

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import pyarrow.parquet as pq
from filelock import FileLock, Timeout
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

LOCK_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"
DF_T = TypeVar("DF_T")


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
    read_fn: Callable[[Path], DF_T],
    write_fn: Callable[[DF_T, Path], None],
    compute_fn: Callable[[DF_T], DF_T],
    do_overwrite: bool = False,
    out_fp_checker: Callable[[Path], bool] = default_file_checker,
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
            return a dataframe of (generic) type DF_T. Ideally, this read function can make use of lazy
            loading to further accelerate unnecessary reads when resuming from intermediate cached steps.
        write_fn: Function that writes the dataframe to a file. This must take as input a dataframe of
            (generic) type DF_T and a Path object, and will write the dataframe to that file.
        compute_fn: A function that transform the dataframe, which must take as input a dataframe of (generic)
            type DF_T and return a dataframe of (generic) type DF_T.
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
        >>> assert not out_fp.is_file() # Out file should not be created when the process crashes

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
        st_time = datetime.now()
        logger.info(f"Reading input dataframe from {in_fp}")
        df = read_fn(in_fp)
        logger.info("Read dataset")
        df = compute_fn(df)
        logger.info(f"Writing final output to {out_fp}")
        write_fn(df, out_fp)
        logger.info(f"Succeeded in {datetime.now() - st_time}")
        return True
    finally:
        lock.release()
        lock_fp.unlink()


def shuffle_shards(shards: list[str], cfg: DictConfig) -> list[str]:
    """Shuffle the shards in a deterministic, pseudo-random way based on the worker ID in the configuration.

    Args:
        shards: The list of shards to shuffle.
        cfg: The configuration dictionary for the overall pipeline. Should (possibly) contain the following
            keys (some are optional, as marked below):
            - `worker` (optional): The worker ID for the MR worker; this is also used to seed the
              randomization process. If not provided, the randomization process will be unseeded.

    Returns:
        The shuffled list of shards.

    Examples:
        >>> shards = ["train/0", "train/1", "tuning", "held_out"]
        >>> shuffle_shards(shards, DictConfig({"worker": 1}))
        ['train/1', 'held_out', 'tuning', 'train/0']
        >>> shuffle_shards(shards, DictConfig({"worker": 2}))
        ['tuning', 'held_out', 'train/1', 'train/0']

        It can also shuffle the shards without a worker ID, but the order is then based on the time, which
        is not consistent across runs.
        >>> sorted(shuffle_shards(shards, DictConfig({})))
        ['held_out', 'train/0', 'train/1', 'tuning']

        If the shards aren't unique, it will error
        >>> shards = ["train/0", "train/0", "tuning", "held_out"]
        >>> shuffle_shards(shards, DictConfig({"worker": 1}))
        Traceback (most recent call last):
            ...
        ValueError: Hash collision for shard train/0 with add_str 1!
    """

    if "worker" in cfg:
        add_str = str(cfg["worker"])
    else:
        add_str = str(datetime.now())

    shard_keys = []
    for shard in shards:
        shard_hash = int(hashlib.sha256((add_str + shard).encode("utf-8")).hexdigest(), 16)
        if shard_hash in shard_keys:
            raise ValueError(f"Hash collision for shard {shard} with add_str {add_str}!")
        shard_keys.append(shard_hash)

    return [shard for _, shard in sorted(zip(shard_keys, shards))]


def shard_iterator(
    cfg: DictConfig,
    out_suffix: str = ".parquet",
    in_prefix: str = "",
) -> tuple[list[tuple[Path, Path]], bool]:
    """Returns a list of the shards found in the input directory and their corresponding output directories.

    Args:
        cfg: The configuration dictionary for the overall pipeline. Should (possibly) contain the following
            keys (some are optional, as marked below):
            - `stage_cfg.data_input_dir` (mandatory): The directory containing the input data.
            - `stage_cfg.output_dir` (mandatory): The directory to write the output data.
            - `stage_cfg.train_only` (optional): The prefix of the shards to process (e.g.,
              `"train/"`). If not provided, all shards will be processed.
            - `worker` (optional): The worker ID for the MR worker; this is also used to seed the
              randomization process. If not provided, the randomization process will be unseeded.
        out_suffix: The suffix of the output files. Defaults to ".parquet".
        in_prefix: The prefix of the input files. Defaults to "". This must be a full path component. It can
            end with a slash but even if it doesn't it will be interpreted as a full path component.

    Yields:
        Randomly shuffled pairs of input and output file paths for each shard. The randomization process is
        seeded by the worker ID in ``cfg``, if provided, otherwise it is left unseeded.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...     "code": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        ...     "time": [1, 2, 3, 4, 5, 6, 1, 2, 3],
        ... })
        >>> shards = {"train/0": [1, 2, 3, 4], "train/1": [5, 6, 7], "tuning": [8], "held_out": [9]}
        >>> def write_dfs(input_dir: Path, df: pl.DataFrame=df, shards: dict=shards, sfx: str=".parquet"):
        ...     for shard_name, subject_ids in shards.items():
        ...         df = df.filter(pl.col("subject_id").is_in(subject_ids))
        ...         shard_fp = input_dir / f"{shard_name}{sfx}"
        ...         shard_fp.parent.mkdir(exist_ok=True, parents=True)
        ...         if sfx == ".parquet": df.write_parquet(shard_fp)
        ...         elif sfx == ".csv": df.write_csv(shard_fp)
        ...         else: raise ValueError(f"Unsupported suffix {sfx}")
        ...     return

    By default, this will load all shards in the input directory and write specify their appropriate output
    directories:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     input_dir = root / "data"
        ...     output_dir = root / "output"
        ...     write_dfs(input_dir)
        ...     cfg = DictConfig({
        ...         "stage_cfg": {"data_input_dir": str(input_dir), "output_dir": str(output_dir)},
        ...         "worker": 1,
        ...     })
        ...     fps, includes_only_train = shard_iterator(cfg)
        >>> [(i.relative_to(root), o.relative_to(root)) for i, o in fps]
        [(PosixPath('data/train/1.parquet'),  PosixPath('output/train/1.parquet')),
         (PosixPath('data/held_out.parquet'), PosixPath('output/held_out.parquet')),
         (PosixPath('data/tuning.parquet'),   PosixPath('output/tuning.parquet')),
         (PosixPath('data/train/0.parquet'),  PosixPath('output/train/0.parquet'))]
        >>> includes_only_train
        False

    Different workers will shuffle the shards differently:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     input_dir = root / "data"
        ...     output_dir = root / "output"
        ...     write_dfs(input_dir)
        ...     cfg = DictConfig({
        ...         "stage_cfg": {"data_input_dir": str(input_dir), "output_dir": str(output_dir)},
        ...         "worker": 2,
        ...     })
        ...     fps, includes_only_train = shard_iterator(cfg)
        >>> [(i.relative_to(root), o.relative_to(root)) for i, o in fps]
        [(PosixPath('data/tuning.parquet'),   PosixPath('output/tuning.parquet')),
         (PosixPath('data/held_out.parquet'), PosixPath('output/held_out.parquet')),
         (PosixPath('data/train/1.parquet'),  PosixPath('output/train/1.parquet')),
         (PosixPath('data/train/0.parquet'),  PosixPath('output/train/0.parquet'))]
        >>> includes_only_train
        False

    We can also make it look within a specific input subdir of the data directory and change the output
    suffix. Note that using a specific input subdir is _different_ than requesting it load only train.
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     input_dir = root / "data"
        ...     output_dir = root / "output"
        ...     write_dfs(input_dir)
        ...     cfg = DictConfig({
        ...         "stage_cfg": {"data_input_dir": str(input_dir), "output_dir": str(output_dir)},
        ...         "worker": 1,
        ...     })
        ...     fps, includes_only_train = shard_iterator(cfg, in_prefix="train", out_suffix=".csv")
        >>> [(i.relative_to(root), o.relative_to(root)) for i, o in fps]
        [(PosixPath('data/train/0.parquet'),  PosixPath('output/0.csv')),
         (PosixPath('data/train/1.parquet'),  PosixPath('output/1.csv'))]
        >>> includes_only_train
        False

    We can also make it load only 'train' shards, in the case that there are shards with a valid "train/"
    prefix.
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     input_dir = root / "data"
        ...     output_dir = root / "output"
        ...     write_dfs(input_dir)
        ...     cfg = DictConfig({
        ...         "stage_cfg": {
        ...             "data_input_dir": str(input_dir), "output_dir": str(output_dir),
        ...             "train_only": True,
        ...         },
        ...         "worker": 1,
        ...     })
        ...     fps, includes_only_train = shard_iterator(cfg)
        >>> [(i.relative_to(root), o.relative_to(root)) for i, o in fps]
        [(PosixPath('data/train/1.parquet'),  PosixPath('output/train/1.parquet')),
         (PosixPath('data/train/0.parquet'),  PosixPath('output/train/0.parquet'))]
        >>> includes_only_train
        True

    The train prefix used is precisely `train/` -- other uses of train will not work:
        >>> wrong_pfx_shards = {"train": [1, 2, 3], "train_1": [4, 5, 6], "train-2": [7, 8, 9]}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     input_dir = root / "data"
        ...     output_dir = root / "output"
        ...     write_dfs(input_dir, shards=wrong_pfx_shards)
        ...     cfg = DictConfig({
        ...         "stage_cfg": {
        ...             "data_input_dir": str(input_dir), "output_dir": str(output_dir),
        ...             "train_only": True,
        ...         },
        ...         "worker": 1,
        ...     })
        ...     fps, includes_only_train = shard_iterator(cfg)
        >>> [(i.relative_to(root), o.relative_to(root)) for i, o in fps]
        [(PosixPath('data/train_1.parquet'),  PosixPath('output/train_1.parquet')),
         (PosixPath('data/train-2.parquet'),  PosixPath('output/train-2.parquet')),
         (PosixPath('data/train.parquet'),  PosixPath('output/train.parquet'))]
        >>> includes_only_train
        False

    If there are no such shards, then it loads them all and assumes the filtering will be handled via the
    splits parquet file.
        >>> no_pfx_shards = {"0": [1, 2, 3], "1": [4, 5, 6], "2": [7, 8, 9]}
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     input_dir = root / "data"
        ...     output_dir = root / "output"
        ...     write_dfs(input_dir, shards=no_pfx_shards)
        ...     cfg = DictConfig({
        ...         "stage_cfg": {
        ...             "data_input_dir": str(input_dir), "output_dir": str(output_dir),
        ...             "train_only": True,
        ...         },
        ...         "worker": 1,
        ...     })
        ...     fps, includes_only_train = shard_iterator(cfg)
        >>> [(i.relative_to(root), o.relative_to(root)) for i, o in fps]
        [(PosixPath('data/0.parquet'), PosixPath('output/0.parquet')),
         (PosixPath('data/1.parquet'), PosixPath('output/1.parquet')),
         (PosixPath('data/2.parquet'), PosixPath('output/2.parquet'))]
        >>> includes_only_train
        False

    If it can't find any files, it will error:
        >>> fps, includes_only_train = shard_iterator(cfg)
        Traceback (most recent call last):
            ...
        FileNotFoundError: No shards found in ... with suffix .parquet. Directory contents:...
    """

    input_dir = Path(cfg.stage_cfg.data_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)

    in_suffix = ".parquet"

    if in_prefix:
        input_dir = input_dir / in_prefix

    shards = []
    for p in input_dir.glob(f"**/*{in_suffix}"):
        relative_path = p.relative_to(input_dir)
        shard_name = str(relative_path)
        shard_name = shard_name[: -len(in_suffix)]
        shards.append(shard_name)

    if not shards:
        raise FileNotFoundError(
            f"No shards found in {input_dir} with suffix {in_suffix}. Directory contents: "
            f"{', '.join(str(p.relative_to(input_dir)) for p in input_dir.glob('**/*'))}"
        )

    # We initialize this to False and overwrite it if we find dedicated train shards.
    includes_only_train = False

    train_only = cfg.stage_cfg.get("train_only", None)
    train_shards = [shard_name for shard_name in shards if shard_name.startswith("train/")]
    if train_only and train_shards:
        shards = train_shards
        includes_only_train = True
    elif train_only:
        logger.warning(
            f"train_only={train_only} requested but no dedicated train shards found; processing all shards "
            "and relying on `subject_splits.parquet` for filtering."
        )

    shards = shuffle_shards(shards, cfg)

    logger.info(f"Mapping computation over a maximum of {len(shards)} shards")

    out = []
    for sp in shards:
        in_fp = input_dir / f"{sp}{in_suffix}"
        out_fp = output_dir / f"{sp}{out_suffix}"
        # TODO: Could add checking logic for existence of in_fp and/or out_fp here.
        out.append((in_fp, out_fp))

    return out, includes_only_train


def shard_iterator_by_shard_map(cfg: DictConfig) -> tuple[list[str], bool]:
    """Returns an iterator over shard paths and output paths based on a shard map file, not files on disk.

    Args:
        cfg: The configuration dictionary for the overall pipeline. Should contain the following keys:
            - `shards_map_fp` (mandatory): The file path to the shards map file.
            - `stage_cfg.data_input_dir` (mandatory): The directory containing the input data.
            - `stage_cfg.output_dir` (mandatory): The directory to write the output data.
            - `worker` (optional): The worker ID for the MR worker; this is also used to seed the

    Returns:
        A list of pairs of input and output file paths for each shard, as well as a boolean indicating
        whether the shards are only train shards.

    Raises:
        ValueError: If the `shards_map_fp` key is not present in the configuration.
        FileNotFoundError: If the shard map file is not found at the path specified in the configuration.
        ValueError: If the `train_only` key is present in the configuration.

    Examples:
        >>> shard_iterator_by_shard_map(DictConfig({}))
        Traceback (most recent call last):
            ...
        ValueError: shards_map_fp must be present in the configuration for a map-based shard iterator.
        >>> with tempfile.NamedTemporaryFile() as tmp:
        ...     cfg = DictConfig({"shards_map_fp": tmp.name, "stage_cfg": {"train_only": True}})
        ...     shard_iterator_by_shard_map(cfg)
        Traceback (most recent call last):
            ...
        ValueError: train_only is not supported for this stage.
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tmp = Path(tmp)
        ...     shards_map_fp = tmp / "shards_map.json"
        ...     cfg = DictConfig({"shards_map_fp": shards_map_fp, "stage_cfg": {"train_only": False}})
        ...     shard_iterator_by_shard_map(cfg)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Shard map file not found at ...shards_map.json
        >>> shards = {"train/0": [1, 2, 3, 4], "train/1": [5, 6, 7], "tuning": [8], "held_out": [9]}
        >>> with tempfile.NamedTemporaryFile() as tmp:
        ...     _ = Path(tmp.name).write_text(json.dumps(shards))
        ...     cfg = DictConfig({
        ...         "shards_map_fp": tmp.name,
        ...         "worker": 1,
        ...         "stage_cfg": {"data_input_dir": "data", "output_dir": "output"},
        ...     })
        ...     fps, includes_only_train = shard_iterator_by_shard_map(cfg)
        >>> fps
        [(PosixPath('data/train/1'),  PosixPath('output/train/1.parquet')),
         (PosixPath('data/held_out'), PosixPath('output/held_out.parquet')),
         (PosixPath('data/tuning'),   PosixPath('output/tuning.parquet')),
         (PosixPath('data/train/0'),  PosixPath('output/train/0.parquet'))]
        >>> includes_only_train
        False
    """

    if "shards_map_fp" not in cfg:
        raise ValueError("shards_map_fp must be present in the configuration for a map-based shard iterator.")

    if cfg.stage_cfg.get("train_only", None):
        raise ValueError("train_only is not supported for this stage.")

    shard_map_fp = Path(cfg.shards_map_fp)
    if not shard_map_fp.exists():
        raise FileNotFoundError(f"Shard map file not found at {str(shard_map_fp.resolve())}")

    shards = list(json.loads(shard_map_fp.read_text()).keys())

    input_dir = Path(cfg.stage_cfg.data_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)

    shards = shuffle_shards(shards, cfg)

    logger.info(f"Mapping computation over a maximum of {len(shards)} shards")

    out = []
    for sh in shards:
        in_fp = input_dir / sh
        out_fp = output_dir / f"{sh}.parquet"
        out.append((in_fp, out_fp))

    return out, False
