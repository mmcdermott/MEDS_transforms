"""Basic utilities for parallelizable mapreduces on sharded MEDS datasets with caching and locking."""

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime
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
        out.append(InOutFilePair(in_fp, out_fp))

    return out, includes_only_train


def shard_iterator_by_shard_map(cfg: DictConfig) -> tuple[list[InOutFilePair], bool]:
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
        [InOutFilePair(in_fp=PosixPath('data/train/1'), out_fp=PosixPath('output/train/1.parquet')),
         InOutFilePair(in_fp=PosixPath('data/held_out'), out_fp=PosixPath('output/held_out.parquet')),
         InOutFilePair(in_fp=PosixPath('data/tuning'), out_fp=PosixPath('output/tuning.parquet')),
         InOutFilePair(in_fp=PosixPath('data/train/0'), out_fp=PosixPath('output/train/0.parquet'))]
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
        out.append(InOutFilePair(in_fp, out_fp))

    return out, False
