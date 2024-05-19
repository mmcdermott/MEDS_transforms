#!/usr/bin/env python

import json
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.sharding import shard_patients
from MEDS_polars_functions.utils import hydra_loguru_init


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Extracts the set of unique patients from the raw data and splits/shards them and saves the result."""

    hydra_loguru_init()

    logger.info(f"Starting patient splitting and sharding with config:\n{cfg.pretty()}")

    raw_cohort_dir = Path(cfg.raw_cohort_dir)
    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)
    subsharded_input_files = list(raw_cohort_dir.glob("sub_sharded/*/*.parquet"))

    if subsharded_input_files:
        logger.info("Subsharded input files found. Using them.")
        input_path = raw_cohort_dir / "sub_sharded" / "*" / "*.parquet"
    elif list(raw_cohort_dir.glob("raw/*.parquet")):
        logger.info("No subsharded input files found. Using raw input files.")
        input_path = raw_cohort_dir / "raw" / "*.parquet"
    else:
        raise FileNotFoundError("No input files found in either 'raw' or 'sub_sharded' directories.")

    if cfg.external_splits_json_fp:
        external_splits_json_fp = Path(cfg.external_splits_json_fp)
        if not external_splits_json_fp.exists():
            raise FileNotFoundError(f"External splits JSON file not found at {external_splits_json_fp}")

        logger.info(f"Reading external splits from {cfg.external_splits_json_fp}")
        external_splits = json.loads(external_splits_json_fp.read_text())

        size_strs = ", ".join(f"{k}: {len(v)}" for k, v in external_splits.items())
        logger.info(f"Loaded external splits of size: {size_strs}")

    logger.info(f"Reading patient IDs from {input_path}")
    patient_ids = (
        pl.scan_parquet(input_path).select(pl.col("patient_id").unique()).collect(streaming=True).to_list()
    )

    logger.info(f"Found {len(patient_ids)} unique patient IDs. Sharding and splitting")
    sharded_patients = shard_patients(
        patients=patient_ids,
        external_splits=external_splits,
        split_fracs_dict=cfg.split_fracs,
        n_patients_per_shard=cfg.n_patients_per_shard,
        seed=cfg.seed,
    )

    logger.info(f"Writing sharded patients to {MEDS_cohort_dir}")
    MEDS_cohort_dir.mkdir(parents=True, exist_ok=True)
    out_fp = MEDS_cohort_dir / "splits.json"
    out_fp.write_text(json.dumps(sharded_patients))
    logger.info(f"Done writing sharded patients to {out_fp}")


if __name__ == "__main__":
    main()
