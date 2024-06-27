#!/usr/bin/env python
from functools import partial
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.mapreduce.mapper import map_over, shard_iterator

pl.enable_string_cache()


def read_fn(sp_dir: Path, unique_by: list[str] | str | None) -> pl.LazyFrame:
    files_to_read = list(sp_dir.glob("**/*.parquet"))

    if not files_to_read:
        raise FileNotFoundError(f"No files found in {sp_dir}/**/*.parquet.")

    file_strs = "\n".join(f"  - {str(fp.resolve())}" for fp in files_to_read)
    logger.info(f"Reading {len(files_to_read)} files:\n{file_strs}")

    dfs = [pl.scan_parquet(fp, glob=False) for fp in files_to_read]
    df = pl.concat(dfs, how="diagonal_relaxed")

    match unique_by:
        case None:
            pass
        case "*":
            df = df.unique(maintain_order=False)
        case list() if len(unique_by) == 0 and all(isinstance(u, str) for u in unique_by):
            subset = []
            for u in unique_by:
                if u in df.columns:
                    subset.append(u)
                else:
                    logger.warning(f"Column {u} not found in dataframe. Omitting from unique-by subset.")
            df = df.unique(maintain_order=False, subset=subset)
        case _:
            raise ValueError(f"Invalid unique_by value: {unique_by}")

    return df.sort(by=["patient_id", "timestamp"], multithreaded=False)


config_yaml = files("MEDS_polars_functions").joinpath("configs/extraction.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Merges the patient sub-sharded events into a single parquet file per patient shard."""

    map_over(
        cfg,
        read_fn=partial(read_fn, unique_by=cfg.stage_cfg.get("unique_by", None)),
        shard_iterator_fntr=partial(shard_iterator, in_suffix=""),
    )


if __name__ == "__main__":
    main()
