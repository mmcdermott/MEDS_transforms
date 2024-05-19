"""Core utilities for MEDS pipelines built with these tools."""

import os
from pathlib import Path

import hydra
import polars as pl
from loguru import logger as log


def hydra_loguru_init() -> None:
    """Adds loguru output to the logs that hydra scrapes.

    Must be called from a hydra main!
    """
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.add(os.path.join(hydra_path, "main.log"))


def write_lazyframe(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)
