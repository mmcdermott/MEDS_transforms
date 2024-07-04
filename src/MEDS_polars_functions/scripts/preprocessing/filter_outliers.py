#!/usr/bin/env python
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_polars_functions.mapreduce.mapper import map_over
from MEDS_polars_functions.transforms.filter_measurements import filter_outliers_fntr

config_yaml = files("MEDS_polars_functions").joinpath("configs/preprocess.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """TODO."""

    code_metadata = pl.read_parquet(
        Path(cfg.stage_cfg.metadata_input_dir) / "code_metadata.parquet", use_pyarrow=True
    )
    compute_fn = filter_outliers_fntr(cfg.stage_cfg, code_metadata)

    map_over(cfg, compute_fn=compute_fn)


if __name__ == "__main__":
    main()
