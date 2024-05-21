#!/usr/bin/env python

import json
import random
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.filter_patients import (
    filter_patients_by_num_events,
    filter_patients_by_num_measurements,
)
from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe


@hydra.main(version_base=None, config_path="configs", config_name="preprocess")
def main(cfg: DictConfig):
    """TODO."""

    hydra_loguru_init()

    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)
    output_dir = Path(cfg.output_data_dir)

    shards = json.loads((MEDS_cohort_dir / "splits.json").read_text())

    final_cohort_dir = MEDS_cohort_dir / "final_cohort"

    patient_splits = list(shards.keys())
    random.shuffle(patient_splits)

    filtered_patients_dir = output_dir / "patients_above_length_threshold"

    compute_fns = []
    if cfg.min_measurements_per_patient:
        logger.info(
            f"Filtering patients with fewer than {cfg.min_measurements_per_patient} measurements "
            "(observations of any kind)."
        )
        compute_fns.append(
            partial(
                filter_patients_by_num_measurements,
                min_measurements_per_patient=cfg.min_measurements_per_patient,
            )
        )
    if cfg.min_events_per_patient:
        logger.info(
            f"Filtering patients with fewer than {cfg.min_events_per_patient} events (unique timepoints)."
        )
        compute_fns.append(
            partial(filter_patients_by_num_events, min_events_per_patient=cfg.min_events_per_patient)
        )

    for sp in patient_splits:
        in_fp = final_cohort_dir / f"{sp}.parquet"
        out_fp = filtered_patients_dir / f"{sp}.parquet"

        logger.info(f"Filtering {str(in_fp.resolve())} into {str(out_fp.resolve())}")

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

    logger.info("Filtered patients.")


if __name__ == "__main__":
    main()
