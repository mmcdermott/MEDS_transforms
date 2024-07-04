#!/usr/bin/env python
from importlib.resources import files

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapreduce.mapper import map_over
from MEDS_polars_functions.transforms.time_derived_measurements import (
    add_new_events_fntr,
    age_fntr,
    time_of_day_fntr,
)
from MEDS_polars_functions.utils import hydra_loguru_init

INFERRED_STAGE_KEYS = {"is_metadata", "data_input_dir", "metadata_input_dir", "output_dir"}


config_yaml = files("MEDS_polars_functions").joinpath("configs/preprocess.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique timestamp."""

    hydra_loguru_init()

    compute_fns = []
    # We use the raw stages object as the induced `stage_cfg` has extra properties like the input and output
    # directories.
    for feature_name, feature_cfg in cfg.stage_cfg.items():
        match feature_name:
            case "age":
                compute_fns.append(add_new_events_fntr(age_fntr(feature_cfg)))
            case "time_of_day":
                compute_fns.append(add_new_events_fntr(time_of_day_fntr(feature_cfg)))
            case str() if feature_name in INFERRED_STAGE_KEYS:
                continue
            case _:
                raise ValueError(f"Unknown time-derived measurement: {feature_name}")

        logger.info(f"Adding {feature_name} via config: {OmegaConf.to_yaml(feature_cfg)}")

    map_over(cfg, compute_fn=tuple(compute_fns))

    logger.info("Added time-derived measurements.")


if __name__ == "__main__":
    main()
