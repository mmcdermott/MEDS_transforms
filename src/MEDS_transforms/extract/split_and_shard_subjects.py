#!/usr/bin/env python
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms.extract import CONFIG_YAML
from MEDS_transforms.utils import stage_init

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


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Extracts the set of unique subjects from the raw data and splits/shards them and saves the result.

    This stage splits the subjects into training, tuning, and held-out sets, and further splits those sets
    into shards.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file. It cannot be overwritten
    directly on the command line, but can be overwritten implicitly by overwriting components of the
    `stage_configs.split_and_shard_subjects` key.

    Args:
        stage_configs.split_and_shard_subjects.n_subjects_per_shard: The maximum number of subjects to include
            in any shard. Realized shards will not necessarily have this many subjects, though they will never
            exceed this number. Instead, the number of shards necessary to include all subjects in a split
            such that no shard exceeds this number will be calculated, then the subjects will be evenly,
            randomly split amongst those shards so that all shards within a split have approximately the same
            number of patietns.
        stage_configs.split_and_shard_subjects.external_splits_json_fp: The path to a json file containing any
            pre-defined splits for specialty held-out test sets beyond the IID held out set that will be
            produced (e.g., for prospective datasets, etc.).
        stage_configs.split_and_shard_subjects.split_fracs: The fraction of subjects to include in the IID
            training, tuning, and held-out sets. Split fractions can be changed for the default names by
            adding a hydra-syntax command line argument for the nested name; e.g., `split_fracs.train=0.7
            split_fracs.tuning=0.1 split_fracs.held_out=0.2`. A split can be removed with the `~` override
            Hydra syntax. Similarly, a new split name can be added with the standard Hydra `+` override
            option. E.g., `~split_fracs.held_out +split_fracs.test=0.1`. It is the user's responsibility to
            ensure that split fractions sum to 1.
    """

    subsharded_dir, _, _ = stage_init(cfg)

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info(
        f"Reading event conversion config from {event_conversion_cfg_fp} (needed for subject ID columns)"
    )
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    dfs = []

    default_subject_id_col = event_conversion_cfg.pop("subject_id_col", "subject_id")
    for input_prefix, event_cfgs in event_conversion_cfg.items():
        input_subject_id_column = event_cfgs.get("subject_id_col", default_subject_id_col)

        input_fps = list((subsharded_dir / input_prefix).glob("**/*.parquet"))

        input_fps_strs = "\n".join(f"  - {str(fp.resolve())}" for fp in input_fps)
        logger.info(f"Reading subject IDs from {input_prefix} files:\n{input_fps_strs}")

        for input_fp in input_fps:
            dfs.append(
                pl.scan_parquet(input_fp, glob=False)
                .select(pl.col(input_subject_id_column).alias("subject_id"))
                .unique()
            )

    logger.info(f"Joining all subject IDs from {len(dfs)} dataframes")
    subject_ids = (
        pl.concat(dfs, how="vertical_relaxed")
        .select(pl.col("subject_id").drop_nulls().drop_nans().unique())
        .collect(streaming=True)["subject_id"]
        .to_numpy(use_pyarrow=True)
    )

    logger.info(f"Found {len(subject_ids)} unique subject IDs of type {subject_ids.dtype}")

    if cfg.stage_cfg.external_splits_json_fp:
        external_splits_json_fp = Path(cfg.stage_cfg.external_splits_json_fp)
        if not external_splits_json_fp.exists():
            raise FileNotFoundError(f"External splits JSON file not found at {external_splits_json_fp}")

        logger.info(f"Reading external splits from {str(external_splits_json_fp.resolve())}")
        external_splits = json.loads(external_splits_json_fp.read_text())

        size_strs = ", ".join(f"{k}: {len(v)}" for k, v in external_splits.items())
        logger.info(f"Loaded external splits of size: {size_strs}")
    else:
        external_splits = None

    logger.info("Sharding and splitting subjects")

    sharded_subjects = shard_subjects(
        subjects=subject_ids,
        external_splits=external_splits,
        split_fracs_dict=cfg.stage_cfg.split_fracs,
        n_subjects_per_shard=cfg.stage_cfg.n_subjects_per_shard,
        seed=cfg.seed,
    )

    shards_map_fp = Path(cfg.shards_map_fp)
    logger.info(f"Writing sharded subjects to {str(shards_map_fp.resolve())}")
    shards_map_fp.parent.mkdir(parents=True, exist_ok=True)
    shards_map_fp.write_text(json.dumps(sharded_subjects))
    logger.info("Done writing sharded subjects")


if __name__ == "__main__":  # pragma: no cover
    main()
