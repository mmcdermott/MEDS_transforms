"""Basic utilities for parallelizable mapreduces on sharded MEDS datasets with caching and locking."""

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class InOutFilePair(NamedTuple):
    in_fp: Path
    out_fp: Path


SHARD_ITR_FNTR_T = Callable[[DictConfig, str, str], tuple[list[InOutFilePair], bool]]


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
        ValueError: Shards must be unique, but found duplicates: train/0
    """

    if len(shards) != len(set(shards)):
        duplicates = sorted({shard for shard in shards if shards.count(shard) > 1})
        raise ValueError(f"Shards must be unique, but found duplicates: {', '.join(duplicates)}")

    add_str = str(cfg.get("worker", datetime.now(tz=UTC)))

    def hash_fn(shard: str) -> int:
        return int(hashlib.sha256((add_str + shard).encode("utf-8")).hexdigest(), 16)

    return sorted(shards, key=hash_fn)


def shard_iterator(
    cfg: DictConfig,
    out_suffix: str = ".parquet",
    in_prefix: str = "",
) -> tuple[list[InOutFilePair], bool]:
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
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ...         "code": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        ...         "time": [1, 2, 3, 4, 5, 6, 1, 2, 3],
        ...     }
        ... )
        >>> shards = {"train/0": [1, 2, 3, 4], "train/1": [5, 6, 7], "tuning": [8], "held_out": [9]}
        >>> def write_dfs(
        ...     input_dir: Path, df: pl.DataFrame = df, shards: dict = shards, sfx: str = ".parquet"
        ... ):
        ...     for shard_name, subject_ids in shards.items():
        ...         df = df.filter(pl.col("subject_id").is_in(subject_ids))
        ...         shard_fp = input_dir / f"{shard_name}{sfx}"
        ...         shard_fp.parent.mkdir(exist_ok=True, parents=True)
        ...         if sfx == ".parquet":
        ...             df.write_parquet(shard_fp)
        ...         elif sfx == ".csv":
        ...             df.write_csv(shard_fp)
        ...         else:
        ...             raise ValueError(f"Unsupported suffix {sfx}")
        ...     return

    By default, this will load all shards in the input directory and write specify their appropriate output
    directories:

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     input_dir = root / "data"
        ...     output_dir = root / "output"
        ...     write_dfs(input_dir)
        ...     cfg = DictConfig(
        ...         {
        ...             "stage_cfg": {"data_input_dir": str(input_dir), "output_dir": str(output_dir)},
        ...             "worker": 1,
        ...         }
        ...     )
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
        ...     cfg = DictConfig(
        ...         {
        ...             "stage_cfg": {"data_input_dir": str(input_dir), "output_dir": str(output_dir)},
        ...             "worker": 2,
        ...         }
        ...     )
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
        ...     cfg = DictConfig(
        ...         {
        ...             "stage_cfg": {"data_input_dir": str(input_dir), "output_dir": str(output_dir)},
        ...             "worker": 1,
        ...         }
        ...     )
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
        ...     cfg = DictConfig(
        ...         {
        ...             "stage_cfg": {
        ...                 "data_input_dir": str(input_dir),
        ...                 "output_dir": str(output_dir),
        ...                 "train_only": True,
        ...             },
        ...             "worker": 1,
        ...         }
        ...     )
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
        ...     cfg = DictConfig(
        ...         {
        ...             "stage_cfg": {
        ...                 "data_input_dir": str(input_dir),
        ...                 "output_dir": str(output_dir),
        ...                 "train_only": True,
        ...             },
        ...             "worker": 1,
        ...         }
        ...     )
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
        ...     cfg = DictConfig(
        ...         {
        ...             "stage_cfg": {
        ...                 "data_input_dir": str(input_dir),
        ...                 "output_dir": str(output_dir),
        ...                 "train_only": True,
        ...             },
        ...             "worker": 1,
        ...         }
        ...     )
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
        out.append(InOutFilePair(in_fp, out_fp))

    return out, includes_only_train
