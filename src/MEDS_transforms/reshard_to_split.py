#!/usr/bin/env python
"""Utilities for re-sharding a MEDS cohort to subsharded splits."""

import json
import logging
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from meds import subject_id_field, subject_splits_filepath, time_field
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.extract.split_and_shard_subjects import shard_subjects
from MEDS_transforms.mapreduce import rwlock_wrap, shard_iterator, shuffle_shards
from MEDS_transforms.utils import stage_init, write_lazyframe

logger = logging.getLogger(__name__)


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


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
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


if __name__ == "__main__":  # pragma: no cover
    main()
