"""Transformations for adding time-derived measurements (e.g., a subject's age) to a MEDS dataset."""

import logging
from collections.abc import Callable

import polars as pl
from meds import subject_id_field, time_field
from omegaconf import DictConfig, OmegaConf

from ... import INFERRED_STAGE_KEYS
from .. import Stage
from .age import age_fntr
from .time_of_day import time_of_day_fntr
from .timeline_tokens import timeline_tokens_fntr

logger = logging.getLogger(__name__)


def add_new_events_fntr(fn: Callable[[pl.DataFrame], pl.DataFrame]) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Creates a "meta" functor that computes the input functor on a MEDS shard then combines both dataframes.

    Args:
        fn: The function that computes the new events.

    Returns:
        A function that computes the new events and combines them with the original DataFrame, returning a
        result in proper MEDS sorted order.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 1, 2, 2, 3, 3],
        ...         "time": [
        ...             None,
        ...             datetime(1990, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(1988, 1, 2),
        ...             datetime(2023, 1, 3),
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 1),
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
        │ 1          ┆ 1990-01-01 00:00:00 ┆ DOB    │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ DOB    │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ lab//B │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ dx//1  │
        └────────────┴─────────────────────┴────────┘

    As an example, we'll use the `age_fntr` defined in this module:

        >>> age_cfg = DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "years"})
        >>> age_fn = age_fntr(age_cfg)
        >>> age_fn(df)
        shape: (2, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str  ┆ f32           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE  ┆ 31.001347     │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE  ┆ 35.004169     │
        └────────────┴─────────────────────┴──────┴───────────────┘
        >>> # Now, we'll use the add_new_events functor to add these age events to the original DataFrame.
        >>> add_age_fn = add_new_events_fntr(age_fn)
        >>> add_age_fn(df)
        shape: (10, 4)
        ┌────────────┬─────────────────────┬────────┬───────────────┐
        │ subject_id ┆ time                ┆ code   ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---    ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str    ┆ f32           │
        ╞════════════╪═════════════════════╪════════╪═══════════════╡
        │ 1          ┆ null                ┆ static ┆ null          │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ DOB    ┆ null          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE    ┆ 31.001347     │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A ┆ null          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B ┆ null          │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ DOB    ┆ null          │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE    ┆ 35.004169     │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A ┆ null          │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ lab//B ┆ null          │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ dx//1  ┆ null          │
        └────────────┴─────────────────────┴────────┴───────────────┘
    """

    def out_fn(df: pl.DataFrame) -> pl.DataFrame:
        new_events = fn(df)
        df = df.with_row_index("__idx")
        new_events = new_events.with_columns(pl.lit(0, dtype=df.schema["__idx"]).alias("__idx"))
        return (
            pl.concat([df, new_events], how="diagonal")
            .sort(by=[subject_id_field, time_field, "__idx"])
            .drop("__idx")
        )

    return out_fn


@Stage.register
def add_time_derived_measurements(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Adds all requested time-derived measurements to a DataFrame.

    Args:
        stage_cfg: The configuration for the time-derived measurements. Recognized time derived functors
            include the following keys:
                - "age": The configuration for the age function.
                - "time_of_day": The configuration for the time of day function.

    Returns:
        A function that adds all requested time-derived measurements to a DataFrame.

    Raises:
        ValueError: If an unrecognized time-derived measurement is requested.

    Examples:
        >>> add_time_derived_measurements(DictConfig({"buzz": {}}))
        Traceback (most recent call last):
            ...
        ValueError: Unknown time-derived measurement: buzz
    """

    map_fns = []
    # We use the raw stages object as the induced `stage_cfg` has extra properties like the input and output
    # directories.
    for feature_name, feature_cfg in stage_cfg.items():
        match feature_name:
            case "age":
                map_fns.append(add_new_events_fntr(age_fntr(feature_cfg)))
            case "time_of_day":
                map_fns.append(add_new_events_fntr(time_of_day_fntr(feature_cfg)))
            case "timeline_tokens":
                map_fns.append(add_new_events_fntr(timeline_tokens_fntr(feature_cfg)))
            case str() if feature_name in INFERRED_STAGE_KEYS:
                continue
            case _:
                raise ValueError(f"Unknown time-derived measurement: {feature_name}")

        logger.info(f"Adding {feature_name} via config: {OmegaConf.to_yaml(feature_cfg)}")

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        for map_fn in map_fns:
            df = map_fn(df)
        return df

    return fn
