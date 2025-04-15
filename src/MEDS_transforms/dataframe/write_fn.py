from collections.abc import Callable
from pathlib import Path

import polars as pl

from .types import DF_T

WRITE_FN_T = Callable[[DF_T, Path], None]


def write_df(df: DF_T, out_fp: Path) -> None:
    """A generic helper to write a dataframe, either lazy or eager, to a parquet file."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_fp, use_pyarrow=True)
