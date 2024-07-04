#!/usr/bin/env python
from functools import partial
from importlib.resources import files

import hydra
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.mapreduce.mapper import map_over
from MEDS_polars_functions.transforms.filter_patients_by_length import (
    filter_patients_by_num_events,
    filter_patients_by_num_measurements,
)
from MEDS_polars_functions.utils import hydra_loguru_init

config_yaml = files("MEDS_polars_functions").joinpath("configs/preprocess.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """TODO."""

    hydra_loguru_init()

    compute_fns = []
    if cfg.stage_cfg.min_measurements_per_patient:
        logger.info(
            f"Filtering patients with fewer than {cfg.stage_cfg.min_measurements_per_patient} measurements "
            "(observations of any kind)."
        )
        compute_fns.append(
            partial(
                filter_patients_by_num_measurements,
                min_measurements_per_patient=cfg.stage_cfg.min_measurements_per_patient,
            )
        )
    if cfg.stage_cfg.min_events_per_patient:
        logger.info(
            f"Filtering patients with fewer than {cfg.stage_cfg.min_events_per_patient} events "
            "(unique timepoints)."
        )
        compute_fns.append(
            partial(
                filter_patients_by_num_events, min_events_per_patient=cfg.stage_cfg.min_events_per_patient
            )
        )

    map_over(cfg, compute_fn=tuple(compute_fns))


if __name__ == "__main__":
    main()
