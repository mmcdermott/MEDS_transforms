#!/usr/bin/env python

import time
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
import polars.selectors as cs
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.mapreduce.mapper import map_over
from MEDS_polars_functions.transforms.code_metadata import mapper_fntr, reducer_fntr
from MEDS_polars_functions.utils import write_lazyframe

config_yaml = files("MEDS_polars_functions").joinpath("configs/extraction.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Computes code metadata."""

    mapper_fn = mapper_fntr(cfg.stage_cfg, cfg.get("code_modifier_columns", None))
    all_out_fps = map_over(cfg, compute_fn=mapper_fn)

    if cfg.worker != 0:
        logger.info("Code metadata mapping completed. Exiting")
        return

    logger.info("Starting reduction process")

    while not all(fp.is_file() for fp in all_out_fps):
        logger.info("Waiting to begin reduction for all files to be written...")
        time.sleep(cfg.polling_time)

    start = datetime.now()
    logger.info("All map shards complete! Starting code metadata reduction computation.")
    reducer_fn = reducer_fntr(cfg.stage_cfg, cfg.get("code_modifier_columns", None))

    reduced = reducer_fn(*[pl.scan_parquet(fp, glob=False) for fp in all_out_fps]).with_columns(
        cs.numeric().shrink_dtype().name.keep()
    )
    logger.debug("For an extraction task specifically, we write out specifically to the cohort dir")
    write_lazyframe(reduced, Path(cfg.cohort_dir) / "code_metadata.parquet")
    logger.info(f"Finished reduction in {datetime.now() - start}")


if __name__ == "__main__":
    main()
