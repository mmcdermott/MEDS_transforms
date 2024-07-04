#!/usr/bin/env python

from functools import partial
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_polars_functions.mapreduce.mapper import map_over
from MEDS_polars_functions.transforms.normalization import normalize

config_yaml = files("MEDS_polars_functions").joinpath("configs/preprocess.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """TODO."""

    code_metadata = pl.read_parquet(
        Path(cfg.stage_cfg.metadata_input_dir) / "code_metadata.parquet", use_pyarrow=True
    ).lazy()
    code_modifiers = cfg.get("code_modifier_columns", None)
    compute_fn = partial(normalize, code_metadata=code_metadata, code_modifiers=code_modifiers)

    map_over(cfg, compute_fn=compute_fn)


if __name__ == "__main__":
    main()
