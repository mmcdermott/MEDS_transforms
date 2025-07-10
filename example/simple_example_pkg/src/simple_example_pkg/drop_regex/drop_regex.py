from collections.abc import Callable

import polars as pl
from meds import DataSchema
from omegaconf import DictConfig

from MEDS_transforms.compute_modes.compute_fn import identity_fn
from MEDS_transforms.stages import Stage


@Stage.register
def drop_regex(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Drops the measurements whose codes match a given regex, or do nothing if no regex is provided.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 2, 3],
        ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        ...     "code": ["ICD10:A01", "ICD9:C03", "ICD9:D04"],
        ... })
        >>> drop_regex(DictConfig({"regex": "ICD10:.*"}))(df)
        shape: (2, 3)
        ┌────────────┬─────────────────────┬──────────┐
        │ subject_id ┆ time                ┆ code     │
        │ ---        ┆ ---                 ┆ ---      │
        │ i64        ┆ datetime[μs]        ┆ str      │
        ╞════════════╪═════════════════════╪══════════╡
        │ 2          ┆ 2020-01-02 00:00:00 ┆ ICD9:C03 │
        │ 3          ┆ 2020-01-03 00:00:00 ┆ ICD9:D04 │
        └────────────┴─────────────────────┴──────────┘
        >>> drop_regex(DictConfig({}))(df)
        shape: (3, 3)
        ┌────────────┬─────────────────────┬───────────┐
        │ subject_id ┆ time                ┆ code      │
        │ ---        ┆ ---                 ┆ ---       │
        │ i64        ┆ datetime[μs]        ┆ str       │
        ╞════════════╪═════════════════════╪═══════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ ICD10:A01 │
        │ 2          ┆ 2020-01-02 00:00:00 ┆ ICD9:C03  │
        │ 3          ┆ 2020-01-03 00:00:00 ┆ ICD9:D04  │
        └────────────┴─────────────────────┴───────────┘
    """

    regex = stage_cfg.get("regex", None)

    if not regex:
        return identity_fn

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.filter(~pl.col(DataSchema.code_name).str.contains(regex))

    return fn
