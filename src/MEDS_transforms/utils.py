"""Core utilities for MEDS pipelines built with these tools."""

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def write_lazyframe(df: pl.LazyFrame, out_fp: Path) -> None:
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_fp, use_pyarrow=True)
