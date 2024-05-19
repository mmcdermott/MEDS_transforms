#!/usr/bin/env python

import random
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init


def read_fn(sp_dir: Path) -> pl.LazyFrame:
    dfs = []
    for parquet_fp in sp_dir.glob("**/*.parquet"):
        dfs.append(pl.scan_parquet(parquet_fp))

    return pl.concat(dfs, how="diagonal").sort(by=["patient_id", "timestamp"])


def write_fn(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)


def identity_fn(df: pl.LazyFrame) -> pl.LazyFrame:
    return df


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Merges the patient sub-sharded events into a single parquet file per patient shard."""

    hydra_loguru_init()

    Path(cfg.raw_cohort_dir)
    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)

    shards = MEDS_cohort_dir / "splits.json"

    logger.info("Starting patient shard merging.")

    patient_subsharded_dir = MEDS_cohort_dir / "patient_sub_sharded_events"
    if not patient_subsharded_dir.exists():
        raise FileNotFoundError(f"Patient sub-sharded directory not found: {patient_subsharded_dir}")

    patient_splits = list(shards.keys())
    random.shuffle(patient_splits)

    for sp in patient_splits:
        in_dir = patient_subsharded_dir / sp
        out_fp = MEDS_cohort_dir / f"{sp}.parquet"

        shard_fps = sorted(list(in_dir.glob("**/*.parquet")))
        shard_fp_strs = [f"  * {str(fp.resolve())}" for fp in shard_fps]
        logger.info(f"Merging {len(shard_fp_strs)} shards into {out_fp}:\n" + "\n".join(shard_fp_strs))
        rwlock_wrap(in_dir, out_fp, read_fn, write_fn, identity_fn)

    logger.info("Output cohort written.")


if __name__ == "__main__":
    main()
