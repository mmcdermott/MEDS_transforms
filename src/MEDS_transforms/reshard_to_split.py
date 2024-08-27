#!/usr/bin/env python
"""Utilities for re-sharding a MEDS cohort to subsharded splits."""

import json
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.extract.split_and_shard_patients import shard_patients
from MEDS_transforms.mapreduce.utils import rwlock_wrap, shard_iterator, shuffle_shards
from MEDS_transforms.utils import stage_init, write_lazyframe


def valid_json_file(fp: Path) -> bool:
    """Check if a file is a valid JSON file."""
    if not fp.is_file():
        return False
    try:
        json.loads(fp.read_text())
        return True
    except json.JSONDecodeError:
        return False


def make_new_shards_fn(df: pl.DataFrame, cfg: DictConfig, stage_cfg: DictConfig) -> dict[str, list[str]]:
    splits_map = defaultdict(list)
    for pt_id, sp in df.iter_rows():
        splits_map[sp].append(pt_id)

    return shard_patients(
        patients=df["patient_id"].to_numpy(),
        n_patients_per_shard=stage_cfg.n_patients_per_shard,
        external_splits=splits_map,
        split_fracs_dict=None,
        seed=cfg.get("seed", 1),
    )


def write_json(d: dict, fp: Path) -> None:
    fp.write_text(json.dumps(d))


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Re-shard a MEDS cohort to in a manner that subdivides patient splits."""

    stage_init(cfg)

    output_dir = Path(cfg.stage_cfg.output_dir)

    splits_file = Path(cfg.input_dir) / "metadata" / "patient_splits.parquet"
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
    while not valid_json_file(shards_fp) and iters < max_iters:
        logger.info(f"Waiting to begin until shards map is written. Iteration {iters}/{max_iters}...")
        time.sleep(cfg.polling_time)
        iters += 1

    new_sharded_splits = json.loads(shards_fp.read_text())

    orig_shards_iter, include_only_train = shard_iterator(cfg, out_suffix="")
    if include_only_train:
        raise ValueError("This stage does not support include_only_train=True")

    orig_shards_iter = [(in_fp, out_fp.relative_to(output_dir)) for in_fp, out_fp in orig_shards_iter]

    new_shards = shuffle_shards(list(new_sharded_splits.keys()), cfg)
    new_shards_iter = [(shard_name, output_dir / f"{shard_name}.parquet") for shard_name in new_shards]

    # Step 1: Sub-sharding stage
    logger.info("Starting sub-sharding")

    for subshard_name, out_fp in new_shards_iter:
        patients = new_sharded_splits[subshard_name]

        def read_fn(input_dir: Path) -> pl.LazyFrame:
            df = None
            logger.info(f"Reading shards for {subshard_name} (file names are in the input sharding scheme):")
            for in_fp, _ in orig_shards_iter:
                logger.info(f"  - {str(in_fp.relative_to(input_dir).resolve())}")
                new_df = pl.scan_parquet(in_fp, glob=False).filter(pl.col("patient_id").is_in(patients))
                if df is None:
                    df = new_df
                else:
                    df = df.merge_sorted(new_df, key="patient_id")
            return df

        def compute_fn(df: list[pl.DataFrame]) -> pl.LazyFrame:
            return df.sort(by=["patient_id", "time"], maintain_order=True, multithreaded=False)

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


if __name__ == "__main__":
    main()
