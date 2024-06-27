#!/usr/bin/env python

from importlib.resources import files

import hydra
from omegaconf import DictConfig

from MEDS_polars_functions.mapreduce.mapper import map_over
from MEDS_polars_functions.transforms.code_metadata import mapper_fntr

config_yaml = files("MEDS_polars_functions").joinpath("configs/extraction.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Computes code metadata."""

    mapper_fn = mapper_fntr(cfg.stage_cfg, cfg.get("code_modifier_columns", None))
    map_over(cfg, compute_fn=mapper_fn)


if __name__ == "__main__":
    main()
