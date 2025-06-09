import polars as pl

from MEDS_transforms.stages import Stage


@Stage.register
def identity_stage(df: pl.LazyFrame) -> pl.LazyFrame:
    """A simple stage that returns the input data unchanged."""
    return df
