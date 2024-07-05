#!/usr/bin/env python

from importlib.resources import files

import hydra
from omegaconf import DictConfig

from MEDS_polars_functions.aggregate_code_metadata import run_map_reduce

config_yaml = files("MEDS_polars_functions").joinpath("configs/extraction.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Computes code metadata."""
    run_map_reduce(cfg)


if __name__ == "__main__":
    main()
