"""Utilities for re-sharding a MEDS cohort to subsharded splits."""

import json
import logging
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
from meds import subject_id_field, subject_splits_filepath, time_field
from omegaconf import DictConfig

from ..mapreduce import rwlock_wrap, shard_iterator, shuffle_shards
from ..stage import MEDS_transforms_stage
from ..utils import stage_init, write_lazyframe

logger = logging.getLogger(__name__)


def shard_subjects(
    subjects: np.ndarray,
    n_subjects_per_shard: int = 50000,
    external_splits: dict[str, Sequence[int]] | None = None,
    split_fracs_dict: dict[str, float] | None = {"train": 0.8, "tuning": 0.1, "held_out": 0.1},
    seed: int = 1,
) -> dict[str, list[int]]:
    """Shard a list of subjects, nested within train/tuning/held-out splits.

    This function takes a list of subjects and shards them into train/tuning/held-out splits, with the shards
    of a consistent size, nested within the splits. The function will also respect external splits, if
    provided, such that mandated splits (such as prospective held out sets or pre-existing, task-specific held
    out sets) are with-held and sharded as separate splits from the IID splits defined by `split_fracs_dict`.
    It returns a dictionary mapping the split and shard names (realized as f"{split}/{shard}") to the list of
    subjects in that shard.

    Args:
        subjects: The list of subjects to shard.
        n_subjects_per_shard: The maximum number of subjects to include in each shard.
        external_splits: The externally defined splits to respect. If provided, the keys of this dictionary
            will be used as split names, and the values as the list of subjects in that split. These
            pre-defined splits will be excluded from IID splits generated by this function, but will be
            sharded like normal. Note that this is largely only appropriate for held-out sets for pre-defined
            tasks or test cases (e.g., prospective tests); training subjects should often still be included in
            the IID splits to maximize the amount of data that can be used for training.
        split_fracs_dict: A dictionary mapping the split name to the fraction of subjects to include in that
            split. Defaults to 80% train, 10% tuning, 10% held-out. This can be None or empty only when
            external splits fully specify the population.
        seed: The random seed to use for shuffling the subjects before seeding and sharding. This is useful
            for ensuring reproducibility.

    Returns:
        A dictionary mapping f"{split}/{shard}" to the list of subjects in that shard. This may include
        overlapping subjects across a subset of these splits, but never across shards within a split. Any
        overlap will solely occur between the an external split and another external split.

    Raises:
        ValueError: If the sum of the split fractions in `split_fracs_dict` is not equal to 1.

    Examples:
        >>> subjects = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)
        >>> shard_subjects(subjects, n_subjects_per_shard=3)
        {'train/0': [9, 4, 8], 'train/1': [2, 1, 10], 'train/2': [6, 5], 'tuning/0': [3], 'held_out/0': [7]}
        >>> shard_subjects(subjects, 3, split_fracs_dict={'train': 0.8, 'tuning': 0.2, 'held_out': None})
        {'train/0': [5, 9, 6], 'train/1': [3, 10, 8], 'train/2': [1, 2], 'tuning/0': [7, 4]}
        >>> shard_subjects(subjects, 3, split_fracs_dict={'train': 0.8, 'held_out': None})
        Traceback (most recent call last):
            ...
        ValueError: The sum of the split fractions must be equal to 1. Got 0.8 through {'train': 0.8}.
        >>> external_splits = {
        ...     'taskA/held_out': np.array([8, 9, 10], dtype=int),
        ...     'taskB/held_out': np.array([10, 8, 9], dtype=int),
        ... }
        >>> shard_subjects(subjects, 3, external_splits)
        {'train/0': [5, 7, 4],
         'train/1': [1, 2],
         'tuning/0': [3],
         'held_out/0': [6],
         'taskA/held_out/0': [8, 9, 10],
         'taskB/held_out/0': [10, 8, 9]}
        >>> shard_subjects(subjects, n_subjects_per_shard=3, split_fracs_dict={'train': 0.5})
        Traceback (most recent call last):
            ...
        ValueError: The sum of the split fractions must be equal to 1. Got 0.5 through {'train': 0.5}.
        >>> shard_subjects([1, 2], n_subjects_per_shard=3)
        Traceback (most recent call last):
            ...
        ValueError: Unable to adjust splits to ensure all splits have at least 1 subject.
        >>> external_splits = {
        ...     'train': np.array([1, 2, 3, 4, 5, 6], dtype=int),
        ...     'test': np.array([7, 8, 9, 10], dtype=int),
        ... }
        >>> shard_subjects(subjects, 6, external_splits, split_fracs_dict=None)
        {'train/0': [1, 2, 3, 4, 5, 6], 'test/0': [7, 8, 9, 10]}
        >>> shard_subjects(subjects, 3, external_splits)
        {'train/0': [5, 1, 3], 'train/1': [2, 6, 4], 'test/0': [10, 7], 'test/1': [8, 9]}
    """

    if external_splits is None:
        external_splits = {}
    else:
        for k in list(external_splits.keys()):
            if not isinstance(external_splits[k], np.ndarray):
                logger.warning(
                    f"External split {k} is not a numpy array and thus type safety is not guaranteed. "
                    f"Attempting to convert to numpy array of dtype {subjects.dtype}."
                )
                external_splits[k] = np.array(external_splits[k], dtype=subjects.dtype)

    subjects = np.unique(subjects)

    # Splitting
    all_external_splits = set().union(*external_splits.values())
    is_in_external_split = np.isin(subjects, list(all_external_splits))
    subject_ids_to_split = subjects[~is_in_external_split]

    splits = external_splits

    if split_fracs_dict is not None and None in split_fracs_dict.values():
        filtered_split_fracs_dict = {k: v for k, v in split_fracs_dict.items() if v is not None}
        logger.info(
            "Ignoring splits with null fraction: "
            + ", ".join(set(split_fracs_dict).difference(filtered_split_fracs_dict))
        )
        split_fracs_dict = filtered_split_fracs_dict

    splits_cover = sum(split_fracs_dict.values()) if split_fracs_dict else 0

    rng = np.random.default_rng(seed)
    if n_subjects := len(subject_ids_to_split):
        if not math.isclose(splits_cover, 1):
            raise ValueError(
                f"The sum of the split fractions must be equal to 1. Got {splits_cover} "
                f"through {split_fracs_dict}."
            )
        split_names_idx = rng.permutation(len(split_fracs_dict))
        split_names = np.array(list(split_fracs_dict.keys()))[split_names_idx]
        split_fracs = np.array([split_fracs_dict[k] for k in split_names])
        split_lens = np.round(split_fracs[:-1] * n_subjects).astype(int)
        split_lens = np.append(split_lens, n_subjects - split_lens.sum())

        if split_lens.min() == 0:
            logger.warning(
                "Some splits are empty. Adjusting splits to ensure all splits have at least 1 subject."
            )
            max_split = split_lens.argmax()
            split_lens[max_split] -= 1
            split_lens[split_lens.argmin()] += 1

        if split_lens.min() == 0:
            raise ValueError("Unable to adjust splits to ensure all splits have at least 1 subject.")

        subjects = rng.permutation(subject_ids_to_split)
        subjects_per_split = np.split(subjects, split_lens.cumsum())

        splits = {**{k: v for k, v in zip(split_names, subjects_per_split)}, **splits}
    else:
        if split_fracs_dict:
            logger.warning(
                "External splits were provided covering all subjects, but split_fracs_dict was not empty. "
                "Ignoring the split_fracs_dict."
            )
        else:
            logger.info("External splits were provided covering all subjects.")

    # Sharding
    final_shards = {}
    for sp, pts in splits.items():
        if len(pts) <= n_subjects_per_shard:
            final_shards[f"{sp}/0"] = pts.tolist()
        else:
            pts = rng.permutation(pts)
            n_pts = len(pts)
            n_shards = int(np.ceil(n_pts / n_subjects_per_shard))
            shards = np.array_split(pts, n_shards)
            for i, shard in enumerate(shards):
                final_shards[f"{sp}/{i}"] = shard.tolist()

    seen = {}

    for k, pts in final_shards.items():
        logger.info(f"Split {k} has {len(pts)} subjects.")

        for kk, v in seen.items():
            shared = set(pts).intersection(v)
            if shared:
                logger.info(f"  - intersects {kk} on {len(shared)} subjects.")

        seen[k] = set(pts)

    return final_shards


def valid_json_file(fp: Path) -> bool:
    """Check if a file is a valid JSON file.

    Args:
        fp: Path to the file.

    Returns:
        True if the file is a valid JSON file, False otherwise.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.json"
        ...     valid_json_file(fp)
        False
        >>> with tempfile.NamedTemporaryFile(suffix=".json") as tmpfile:
        ...     fp = Path(tmpfile.name)
        ...     _ = fp.write_text("foobar not a json file.\tHello, world!")
        ...     valid_json_file(fp)
        False
        >>> with tempfile.NamedTemporaryFile(suffix=".json") as tmpfile:
        ...     fp = Path(tmpfile.name)
        ...     _ = fp.write_text('{"foo": "bar"}')
        ...     valid_json_file(fp)
        True
    """
    if not fp.is_file():
        return False
    try:
        json.loads(fp.read_text())
        return True
    except json.JSONDecodeError:
        return False


def make_new_shards_fn(df: pl.DataFrame, cfg: DictConfig, stage_cfg: DictConfig) -> dict[str, list[str]]:
    """This function creates a new sharding scheme for the MEDS cohort."""
    splits_map = defaultdict(list)
    for pt_id, sp in df.iter_rows():
        splits_map[sp].append(pt_id)

    return shard_subjects(
        subjects=df[subject_id_field].to_numpy(),
        n_subjects_per_shard=stage_cfg.n_subjects_per_shard,
        external_splits=splits_map,
        split_fracs_dict=None,
        seed=cfg.get("seed", 1),
    )


def write_json(d: dict, fp: Path) -> None:
    """Write a dictionary to a JSON file.

    Args:
        d: Dictionary to write.
        fp: Path to the file.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     fp = Path(tmpdir) / "test.json"
        ...     write_json({"foo": "bar"}, fp)
        ...     fp.read_text()
        '{"foo": "bar"}'
    """
    fp.write_text(json.dumps(d))


@MEDS_transforms_stage
def main(cfg: DictConfig):
    """Re-shard a MEDS cohort to in a manner that subdivides subject splits."""

    stage_init(cfg)

    output_dir = Path(cfg.stage_cfg.output_dir)

    splits_file = Path(cfg.input_dir) / subject_splits_filepath
    shards_fp = output_dir / ".shards.json"

    rwlock_wrap(
        splits_file,
        shards_fp,
        partial(pl.read_parquet, use_pyarrow=True),
        write_json,
        partial(make_new_shards_fn, cfg=cfg, stage_cfg=cfg.stage_cfg),
        do_overwrite=cfg.do_overwrite,
        out_fp_checker=valid_json_file,
    )

    max_iters = cfg.get("max_iters", 10)
    iters = 0
    while not valid_json_file(shards_fp) and iters < max_iters:  # pragma: no cover
        logger.info(f"Waiting to begin until shards map is written. Iteration {iters}/{max_iters}...")
        time.sleep(cfg.polling_time)
        iters += 1

    new_sharded_splits = json.loads(shards_fp.read_text())

    if cfg.stage_cfg.get("train_only", False):
        raise ValueError("This stage does not support train_only=True")

    orig_shards_iter, _ = shard_iterator(cfg, out_suffix="")

    orig_shards_iter = [(in_fp, out_fp.relative_to(output_dir)) for in_fp, out_fp in orig_shards_iter]

    new_shards = shuffle_shards(list(new_sharded_splits.keys()), cfg)
    new_shards_iter = [(shard_name, output_dir / f"{shard_name}.parquet") for shard_name in new_shards]

    # Step 1: Sub-sharding stage
    logger.info("Starting sub-sharding")

    for subshard_name, out_fp in new_shards_iter:
        subjects = new_sharded_splits[subshard_name]

        def read_fn(input_dir: Path) -> pl.LazyFrame:
            df = None
            logger.info(f"Reading shards for {subshard_name} (file names are in the input sharding scheme):")
            for in_fp, _ in orig_shards_iter:
                logger.info(f"  - {str(in_fp.relative_to(input_dir).resolve())}")
                new_df = pl.scan_parquet(in_fp, glob=False).filter(pl.col(subject_id_field).is_in(subjects))
                if df is None:
                    df = new_df
                else:
                    df = df.merge_sorted(new_df, key=subject_id_field)
            return df

        def compute_fn(df: list[pl.DataFrame]) -> pl.LazyFrame:
            return df.sort(by=[subject_id_field, time_field], maintain_order=True, multithreaded=False)

        def write_fn(df: pl.LazyFrame, out_fp: Path) -> None:
            write_lazyframe(df, out_fp)

        logger.info(f"Merging sub-shards for {subshard_name} to {str(out_fp.resolve())}")
        rwlock_wrap(
            cfg.stage_cfg.data_input_dir,
            out_fp,
            read_fn,
            write_fn,
            compute_fn,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")
