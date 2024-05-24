#!/usr/bin/env python

import json
import random
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.time_derived_measurements import (
    add_new_events_fntr,
    age_fntr,
    time_of_day_fntr,
)
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe


@hydra.main(version_base=None, config_path="configs", config_name="preprocess")
def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique timestamp."""

    hydra_loguru_init()

    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)
    output_dir = Path(cfg.output_data_dir)

    shards = json.loads((MEDS_cohort_dir / "splits.json").read_text())

    final_cohort_dir = MEDS_cohort_dir / "final_cohort"
    filtered_patients_dir = output_dir / "patients_above_length_threshold"
    with_time_derived_dir = output_dir / "with_time_derived_measurements"

    if filtered_patients_dir.is_dir():
        logger.info(f"Reading data from filtered cohort directory {str(filtered_patients_dir.resolve())}")
        input_dir = filtered_patients_dir
    else:
        logger.info(f"Reading data from raw cohort directory {str(final_cohort_dir.resolve())}")
        input_dir = final_cohort_dir

    patient_splits = list(shards.keys())
    random.shuffle(patient_splits)

    compute_fns = []
    for feature_name, feature_cfg in cfg.time_derived_measurements.items():
        match feature_name:
            case "age":
                compute_fns.append(add_new_events_fntr(age_fntr(feature_cfg)))
            case "time_of_day":
                compute_fns.append(add_new_events_fntr(time_of_day_fntr(feature_cfg)))
            case _:
                raise ValueError(f"Unknown time-derived measurement: {feature_name}")

        logger.info(f"Adding {feature_name} via config: {OmegaConf.to_yaml(feature_cfg)}")

    for sp in patient_splits:
        in_fp = input_dir / f"{sp}.parquet"
        out_fp = with_time_derived_dir / f"{sp}.parquet"

        logger.info(
            f"Adding time derived measurements to {str(in_fp.resolve())} into {str(out_fp.resolve())}"
        )

        rwlock_wrap(
            in_fp,
            out_fp,
            pl.scan_parquet,
            write_lazyframe,
            *compute_fns,
            do_return=False,
            cache_intermediate=False,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info("Added time-derived measurements.")


if __name__ == "__main__":
    main()
