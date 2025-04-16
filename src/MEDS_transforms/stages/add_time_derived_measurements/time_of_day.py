"""A functor to compute time-of-day measurements on a MEDS dataset."""

import logging
from collections.abc import Callable

import polars as pl
from meds import code_field, time_field
from omegaconf import DictConfig

from .utils import unique_events

logger = logging.getLogger(__name__)


def time_of_day_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Create a function that adds the time of day to a DataFrame.

    Args:
        cfg: The configuration for the time of day function. This must contain an "endpoints" key with the
            endpoints (in hours of the day in 24-hour time) for each time of day category and a
            "time_of_day_code" key for the code for the time of day event in the output data.

    Returns:
        A function that adds the time of day to a DataFrame.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 1, 2, 2, 3, 3],
        ...         "time": [
        ...             None,
        ...             datetime(1990, 1, 1, 1, 0),
        ...             datetime(2021, 1, 1, 12, 0),
        ...             datetime(2021, 1, 2, 23, 59),
        ...             datetime(1988, 1, 2, 6, 0),
        ...             datetime(2023, 1, 3, 12, 0),
        ...             datetime(2022, 1, 1, 18, 0),
        ...             datetime(2022, 1, 1, 18, 0),
        ...         ],
        ...         "code": ["static", "DOB", "lab//A", "lab//B", "DOB", "lab//A", "lab//B", "dx//1"],
        ...     },
        ...     schema={"subject_id": pl.UInt32, "time": pl.Datetime, "code": pl.Utf8},
        ... )
        >>> df
        shape: (8, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ subject_id ┆ time                ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ u32        ┆ datetime[μs]        ┆ str    │
        ╞════════════╪═════════════════════╪════════╡
        │ 1          ┆ null                ┆ static │
        │ 1          ┆ 1990-01-01 01:00:00 ┆ DOB    │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ lab//A │
        │ 1          ┆ 2021-01-02 23:59:00 ┆ lab//B │
        │ 2          ┆ 1988-01-02 06:00:00 ┆ DOB    │
        │ 2          ┆ 2023-01-03 12:00:00 ┆ lab//A │
        │ 3          ┆ 2022-01-01 18:00:00 ┆ lab//B │
        │ 3          ┆ 2022-01-01 18:00:00 ┆ dx//1  │
        └────────────┴─────────────────────┴────────┘
        >>> time_of_day_cfg = DictConfig({"time_of_day_code": "time_of_day", "endpoints": [6, 12, 18]})
        >>> time_of_day_fn = time_of_day_fntr(time_of_day_cfg)
        >>> time_of_day_fn(df)
        shape: (6, 3)
        ┌────────────┬─────────────────────┬──────────────────────┐
        │ subject_id ┆ time                ┆ code                 │
        │ ---        ┆ ---                 ┆ ---                  │
        │ u32        ┆ datetime[μs]        ┆ str                  │
        ╞════════════╪═════════════════════╪══════════════════════╡
        │ 1          ┆ 1990-01-01 01:00:00 ┆ time_of_day//[00,06) │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ time_of_day//[12,18) │
        │ 1          ┆ 2021-01-02 23:59:00 ┆ time_of_day//[18,24) │
        │ 2          ┆ 1988-01-02 06:00:00 ┆ time_of_day//[06,12) │
        │ 2          ┆ 2023-01-03 12:00:00 ┆ time_of_day//[12,18) │
        │ 3          ┆ 2022-01-01 18:00:00 ┆ time_of_day//[18,24) │
        └────────────┴─────────────────────┴──────────────────────┘
        >>> time_of_day_fntr(DictConfig({"endpoints": []}))
        Traceback (most recent call last):
            ...
        ValueError: The 'endpoints' key must contain at least one endpoint for time of day categories.
        >>> time_of_day_fntr(DictConfig({"endpoints": [6, 12, 36]}))
        Traceback (most recent call last):
            ...
        ValueError: All endpoints must be between 0 and 24 inclusive. Got: [6, 12, 36]
        >>> time_of_day_fntr(DictConfig({"endpoints": [6, 1.2]}))
        Traceback (most recent call last):
            ...
        ValueError: All endpoints must be integer, whole-hour boundaries, but got: [6, 1.2]
        >>> time_of_day_fntr(DictConfig({"endpoints": [6, 6]}))
        Traceback (most recent call last):
            ...
        ValueError: All endpoints must be unique. Got: [6, 6]
        >>> time_of_day_fntr(DictConfig({"endpoints": [6, 12, 10]}))
        Traceback (most recent call last):
            ...
        ValueError: All endpoints must be in sorted order. Got: [6, 12, 10]
    """
    if not cfg.endpoints:
        raise ValueError("The 'endpoints' key must contain at least one endpoint for time of day categories.")
    if not all(0 <= endpoint <= 24 for endpoint in cfg.endpoints):
        raise ValueError(f"All endpoints must be between 0 and 24 inclusive. Got: {cfg.endpoints}")
    if not all(isinstance(endpoint, int) for endpoint in cfg.endpoints):
        raise ValueError(f"All endpoints must be integer, whole-hour boundaries, but got: {cfg.endpoints}")
    if len(cfg.endpoints) != len(set(cfg.endpoints)):
        raise ValueError(f"All endpoints must be unique. Got: {cfg.endpoints}")
    if cfg.endpoints != sorted(cfg.endpoints):
        raise ValueError(f"All endpoints must be in sorted order. Got: {cfg.endpoints}")

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        hour = pl.col(time_field).dt.hour()

        def tod_code(start: int, end: int) -> str:
            return pl.lit(f"{cfg.time_of_day_code}//[{start:02},{end:02})", dtype=df.schema[code_field])

        start, end = 0, cfg.endpoints[0]
        time_of_day = pl.when(hour < end).then(tod_code(start, end))

        for i in range(1, len(cfg.endpoints)):
            start, end = cfg.endpoints[i - 1], cfg.endpoints[i]

            time_of_day = time_of_day.when((hour >= start) & (hour < end)).then(tod_code(start, end))

        time_of_day = time_of_day.when(hour >= end).then(tod_code(end, 24))
        return unique_events(df).with_columns(time_of_day.alias(code_field))

    return fn
