#!/usr/bin/env python
import json
import random
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapreduce.mapper import rwlock_wrap
from MEDS_polars_functions.transforms.time_derived_measurements import (
    add_new_events_fntr,
    age_fntr,
    time_of_day_fntr,
)
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe

INFERRED_STAGE_KEYS = {"is_metadata", "data_input_dir", "metadata_input_dir", "output_dir"}


config_yaml = files("MEDS_polars_functions").joinpath("configs/preprocess.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique timestamp."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    output_dir = Path(cfg.stage_cfg.output_dir)

    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())
    input_dir = Path(cfg.stage_cfg.data_input_dir)

    logger.info(f"Reading data from {str(input_dir.resolve())}")

    patient_splits = list(shards.keys())
    random.shuffle(patient_splits)

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

    for sp in patient_splits:
        in_fp = input_dir / f"{sp}.parquet"
        out_fp = output_dir / f"{sp}.parquet"

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
