#!/usr/bin/env python

import random
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.event_conversion import convert_to_events
from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init


def write_fn(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)


@hydra.main(version_base=None, config_path="configs", config_name="extraction")
def main(cfg: DictConfig):
    """Converts the sub-sharded or raw data into events which are sharded by patient X input shard."""

    hydra_loguru_init()

    raw_cohort_dir = Path(cfg.raw_cohort_dir)
    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)

    shards = MEDS_cohort_dir / "splits.json"

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info(f"Starting event conversion with config:\n{cfg.pretty()}")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{event_conversion_cfg.pretty()}")

    default_patient_id_col = event_conversion_cfg.pop("patient_id_col", "patient_id")

    patient_subsharded_dir = raw_cohort_dir / "patient_sub_sharded_events"
    OmegaConf.save(event_conversion_cfg, patient_subsharded_dir / "event_conversion_config.yaml")

    patient_splits = list(shards.items())
    random.shuffle(patient_splits)

    event_configs = list(event_conversion_cfg.items())
    random.shuffle(event_configs)

    for sp, patients in patient_splits:
        for input_prefix, event_cfgs in event_configs:
            input_patient_id_column = event_cfgs.get("patient_id_col", default_patient_id_col)

            event_shards = list((raw_cohort_dir / "sub_sharded" / input_prefix).glob("*.parquet"))
            random.shuffle(event_shards)

            for shard_fp in event_shards:
                out_fp = patient_subsharded_dir / sp / input_prefix / shard_fp.name
                logger.info(f"Converting {shard_fp} to events and saving to {out_fp}")

                def compute_fn(df: pl.LazyFrame) -> pl.LazyFrame:
                    if input_patient_id_column != "patient_id":
                        df = (
                            df.with_columns(pl.col(input_patient_id_column).alias("patient_id"))
                            .drop(input_patient_id_column)
                            .filter(pl.col("patient_id").is_in(patients))
                        )

                    try:
                        convert_to_events(df, event_cfgs=event_cfgs, patients=patients)
                    except Exception as e:
                        raise ValueError(
                            f"Error converting {str(shard_fp.resolve())} for {sp}/{input_prefix}: {e}"
                        ) from e

                rwlock_wrap(shard_fp, out_fp, pl.scan_parquet, write_fn, compute_fn)

    logger.info("Subsharded into converted events.")


if __name__ == "__main__":
    main()
