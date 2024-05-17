#!/usr/bin/env python

import random
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init


def convert_to_event(df: pl.LazyFrame, event_cfgs: dict[str, dict[str, str | None]]) -> pl.LazyFrame:
    """Converts a DataFrame of raw data into a DataFrame of events."""

    if not event_cfgs:
        raise ValueError("No event configurations provided.")

    pt_id = pl.col("patient_id")

    event_dfs = []
    for event_name, event_cfg in event_cfgs.items():
        event_exprs = {"patient_id": pt_id}

        code = event_cfg["code"]
        match code:
            case str() if code.startswith("col(") and code.endswith(")"):
                event_exprs["code"] = pl.col(code[4:-1]).cast(pl.Categorical)
            case str():
                event_exprs["code"] = pl.lit(code).cast(pl.Categorical)
            case _:
                raise ValueError(f"Invalid code: {code}")

        ts = event_cfg["timestamp"]
        match ts:
            case str() if ts.startswith("col(") and ts.endswith(")"):
                event_exprs["timestamp"] = pl.col(ts[4:-1]).cast(pl.Datetime)
            case None:
                event_exprs["timestamp"] = pl.lit(None, dtype=pl.Datetime)
            case _:
                raise ValueError(f"Invalid timestamp: {ts}")

        for k, v in event_cfg.get("event_columns", {}).items():
            if k in event_exprs:
                raise KeyError(f"Event column name {k} conflicts with core column name.")
            elif k not in df.schema:
                raise KeyError(f"Event column name {k} not found in DataFrame schema.")

            col = pl.col(v)
            if df.schema[k] == pl.Utf8:
                col = col.cast(pl.Categorical)

            event_exprs[k] = col

        event_dfs.append(df.select(**event_exprs).unique())

    return pl.concat(event_dfs, how="diagonal")


def filter_and_convert[
    PT_ID_T
](df: pl.LazyFrame, event_cfgs: dict[str, dict[str, str | None]], patients: list[PT_ID_T]) -> pl.LazyFrame:
    """Filters the DataFrame and converts it into events."""

    return convert_to_event(df.filter(pl.col("patient_id").isin(patients)), event_cfgs)


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

    patient_subsharded_dir = raw_cohort_dir / "patient_sub_sharded_events"
    OmegaConf.save(event_conversion_cfg, patient_subsharded_dir / "event_conversion_config.yaml")

    patient_splits = list(shards.items())
    random.shuffle(patient_splits)

    event_configs = list(event_conversion_cfg.items())
    random.shuffle(event_configs)

    for sp, patients in patient_splits:
        for input_prefix, event_cfgs in event_configs:
            event_shards = list((raw_cohort_dir / "sub_sharded" / input_prefix).glob("*.parquet"))

            compute_fn = partial(filter_and_convert, event_cfgs=event_cfgs, patients=patients)

            random.shuffle(event_shards)
            for shard_fp in event_shards:
                out_fp = patient_subsharded_dir / sp / input_prefix / shard_fp.name
                logger.info(f"Converting {shard_fp} to events and saving to {out_fp}")

                rwlock_wrap(shard_fp, out_fp, pl.scan_parquet, write_fn, compute_fn)

    logger.info("Subsharded into converted events.")


if __name__ == "__main__":
    main()
