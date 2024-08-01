#!/usr/bin/env python
"""Transformations for adding time-derived measurements (e.g., a patient's age) to a MEDS dataset."""
from collections.abc import Callable

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.utils import hydra_loguru_init

pl.enable_string_cache()


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
        ...         "patient_id": [1, 1, 1, 1, 2, 2, 3, 3],
        ...         "timestamp": [
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
        ...     schema={"patient_id": pl.UInt32, "timestamp": pl.Datetime, "code": pl.Categorical},
        ... )
        >>> df
        shape: (8, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ patient_id ┆ timestamp           ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ u32        ┆ datetime[μs]        ┆ cat    │
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
        >>> # As an example, we'll use the age functor defined elsewhere in this module.
        >>> age_cfg = DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "years"})
        >>> age_fn = age_fntr(age_cfg)
        >>> age_fn(df)
        shape: (2, 4)
        ┌────────────┬─────────────────────┬──────┬─────────────────┐
        │ patient_id ┆ timestamp           ┆ code ┆ numerical_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---             │
        │ u32        ┆ datetime[μs]        ┆ cat  ┆ f64             │
        ╞════════════╪═════════════════════╪══════╪═════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE  ┆ 31.001347       │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE  ┆ 35.00417        │
        └────────────┴─────────────────────┴──────┴─────────────────┘
        >>> # Now, we'll use the add_new_events functor to add these age events to the original DataFrame.
        >>> add_age_fn = add_new_events_fntr(age_fn)
        >>> add_age_fn(df)
        shape: (10, 4)
        ┌────────────┬─────────────────────┬────────┬─────────────────┐
        │ patient_id ┆ timestamp           ┆ code   ┆ numerical_value │
        │ ---        ┆ ---                 ┆ ---    ┆ ---             │
        │ u32        ┆ datetime[μs]        ┆ cat    ┆ f64             │
        ╞════════════╪═════════════════════╪════════╪═════════════════╡
        │ 1          ┆ null                ┆ static ┆ null            │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ DOB    ┆ null            │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE    ┆ 31.001347       │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A ┆ null            │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B ┆ null            │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ DOB    ┆ null            │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE    ┆ 35.00417        │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A ┆ null            │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ lab//B ┆ null            │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ dx//1  ┆ null            │
        └────────────┴─────────────────────┴────────┴─────────────────┘
    """

    def out_fn(df: pl.DataFrame) -> pl.DataFrame:
        new_events = fn(df)
        df = df.with_row_index("__idx")
        new_events = new_events.with_columns(pl.lit(0, dtype=df.schema["__idx"]).alias("__idx"))
        return (
            pl.concat([df, new_events], how="diagonal")
            .sort(by=["patient_id", "timestamp", "__idx"])
            .drop("__idx")
        )

    return out_fn


TIME_DURATION_UNITS = {
    "seconds": (["s", "sec", "secs", "second", "seconds"], 1),
    "minutes": (["m", "min", "mins", "minute", "minutes"], 60),
    "hours": (["h", "hr", "hrs", "hour", "hours"], 60 * 60),
    "days": (["d", "day", "days"], 60 * 60 * 24),
    "weeks": (["w", "wk", "wks", "week", "weeks"], 60 * 60 * 24 * 7),
    "months": (["mo", "mos", "month", "months"], 60 * 60 * 24 * 30.436875),
    "years": (["y", "yr", "yrs", "year", "years"], 60 * 60 * 24 * 365.2422),
}


def normalize_time_unit(unit: str) -> tuple[str, float]:
    """Normalize a time unit string to a canonical form and return the number of seconds in that unit.

    Note that this function is designed for computing _approximate_ time durations over long periods, not
    canonical, local calendar time durations. E.g., a "month" is not a fixed number of seconds, but this
    function will return the average number of seconds in a month, accounting for leap years.

    TODO: consider replacing this function with the use of https://github.com/wroberts/pytimeparse

    Args:
        unit: The input unit to normalize.

    Returns:
        A tuple containing the canonical unit and the number of seconds in that unit.

    Raises:
        ValueError: If the input unit is not recognized.

    Examples:
        >>> normalize_time_unit("s")
        ('seconds', 1)
        >>> normalize_time_unit("min")
        ('minutes', 60)
        >>> normalize_time_unit("hours")
        ('hours', 3600)
        >>> normalize_time_unit("day")
        ('days', 86400)
        >>> normalize_time_unit("wks")
        ('weeks', 604800)
        >>> normalize_time_unit("month")
        ('months', 2629746.0)
        >>> normalize_time_unit("years")
        ('years', 31556926.080000002)
        >>> normalize_time_unit("fortnight")
        Traceback (most recent call last):
            ...
        ValueError: Unknown time unit 'fortnight'. Valid units include:
          * seconds: s, sec, secs, second, seconds
          * minutes: m, min, mins, minute, minutes
          * hours: h, hr, hrs, hour, hours
          * days: d, day, days
          * weeks: w, wk, wks, week, weeks
          * months: mo, mos, month, months
          * years: y, yr, yrs, year, years
    """
    for canonical_unit, (aliases, seconds) in TIME_DURATION_UNITS.items():
        if unit in aliases:
            return canonical_unit, seconds

    valid_unit_lines = []
    for canonical, (aliases, _) in TIME_DURATION_UNITS.items():
        valid_unit_lines.append(f"  * {canonical}: {', '.join(aliases)}")
    valid_units_str = "\n".join(valid_unit_lines)
    raise ValueError(f"Unknown time unit '{unit}'. Valid units include:\n{valid_units_str}")


def age_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Create a function that adds a patient's age to a DataFrame.

    Args:
        cfg: The configuration for the age function. This must contain the following mandatory keys:
            - "DOB_code": The code for the date of birth event in the raw data.
            - "age_code": The code for the age event in the output data.
            - "age_unit": The unit for the age event when converted to a numeric value in the output data.

    Returns:
        A function that returns the to-be-added "age" events with the patient's age for all input events with
        unique, non-null timestamps in the data, for all patients who have an observed date of birth. It does
        not add an event for timestamps that are equal to the date of birth.

    Raises:
        ValueError: If the input unit is not recognized.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 1, 1, 2, 2, 3, 3],
        ...         "timestamp": [
        ...             None,
        ...             datetime(1990, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(1988, 1, 2),
        ...             datetime(2023, 1, 3),
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 1),
        ...         ],
        ...         "code": ["static", "DOB", "lab//A", "lab//B", "rx", "DOB", "lab//A", "lab//B", "dx//1"],
        ...     },
        ...     schema={"patient_id": pl.UInt32, "timestamp": pl.Datetime, "code": pl.Categorical},
        ... )
        >>> df
        shape: (9, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ patient_id ┆ timestamp           ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ u32        ┆ datetime[μs]        ┆ cat    │
        ╞════════════╪═════════════════════╪════════╡
        │ 1          ┆ null                ┆ static │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ DOB    │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ rx     │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ DOB    │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ lab//B │
        │ 3          ┆ 2022-01-01 00:00:00 ┆ dx//1  │
        └────────────┴─────────────────────┴────────┘
        >>> age_cfg = DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "years"})
        >>> age_fn = age_fntr(age_cfg)
        >>> age_fn(df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────┬─────────────────┐
        │ patient_id ┆ timestamp           ┆ code ┆ numerical_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---             │
        │ u32        ┆ datetime[μs]        ┆ cat  ┆ f64             │
        ╞════════════╪═════════════════════╪══════╪═════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE  ┆ 31.001347       │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ AGE  ┆ 31.004084       │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE  ┆ 35.00417        │
        └────────────┴─────────────────────┴──────┴─────────────────┘
        >>> age_cfg = DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "scores"})
        >>> age_fn = age_fntr(age_cfg)
        Traceback (most recent call last):
            ...
        ValueError: Unknown time unit 'scores'. Valid units include:
        ...
    """

    canonical_unit, seconds_in_unit = normalize_time_unit(cfg.age_unit)
    microseconds_in_unit = int(1e6) * seconds_in_unit

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        dob_expr = pl.when(pl.col("code") == cfg.DOB_code).then(pl.col("timestamp")).min().over("patient_id")
        age_expr = (pl.col("timestamp") - dob_expr).dt.total_microseconds() / microseconds_in_unit

        return (
            df.drop_nulls(subset=["timestamp"])
            .unique(subset=["patient_id", "timestamp"], maintain_order=True)
            .select(
                "patient_id",
                "timestamp",
                pl.lit(cfg.age_code, dtype=df.schema["code"]).alias("code"),
                age_expr.alias("numerical_value"),
            )
            .drop_nulls(subset=["numerical_value"])
            .filter(pl.col("numerical_value") > 0)
        )

    return fn


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
        ...         "patient_id": [1, 1, 1, 1, 2, 2, 3, 3],
        ...         "timestamp": [
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
        ...     schema={"patient_id": pl.UInt32, "timestamp": pl.Datetime, "code": pl.Categorical},
        ... )
        >>> df
        shape: (8, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ patient_id ┆ timestamp           ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ u32        ┆ datetime[μs]        ┆ cat    │
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
        │ patient_id ┆ timestamp           ┆ code                 │
        │ ---        ┆ ---                 ┆ ---                  │
        │ u32        ┆ datetime[μs]        ┆ cat                  │
        ╞════════════╪═════════════════════╪══════════════════════╡
        │ 1          ┆ 1990-01-01 01:00:00 ┆ time_of_day//[00,06) │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ time_of_day//[12,18) │
        │ 1          ┆ 2021-01-02 23:59:00 ┆ time_of_day//[18,24) │
        │ 2          ┆ 1988-01-02 06:00:00 ┆ time_of_day//[06,12) │
        │ 2          ┆ 2023-01-03 12:00:00 ┆ time_of_day//[12,18) │
        │ 3          ┆ 2022-01-01 18:00:00 ┆ time_of_day//[18,24) │
        └────────────┴─────────────────────┴──────────────────────┘
    """
    if not cfg.endpoints:
        raise ValueError("The 'endpoints' key must contain at least one endpoint for time of day categories.")
    if not all(0 <= endpoint <= 24 for endpoint in cfg.endpoints):
        raise ValueError(f"All endpoints must be between 0 and 24 inclusive. Got: {cfg.endpoints}")
    if not all(isinstance(endpoint, int) for endpoint in cfg.endpoints):
        raise ValueError(f"All endpoints must be integer, whole-hour boundaries, but got: {cfg.endpoints}")
    if len(cfg.endpoints) != len(set(cfg.endpoints)) or cfg.endpoints != sorted(cfg.endpoints):
        raise ValueError(f"All endpoints must be unique and in sorted order. Got: {cfg.endpoints}")

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        hour = pl.col("timestamp").dt.hour()

        def tod_code(start: int, end: int) -> str:
            return pl.lit(f"{cfg.time_of_day_code}//[{start:02},{end:02})", dtype=df.schema["code"])

        start, end = 0, cfg.endpoints[0]
        time_of_day = pl.when(hour < end).then(tod_code(start, end))

        for i in range(1, len(cfg.endpoints)):
            start, end = cfg.endpoints[i - 1], cfg.endpoints[i]

            time_of_day = time_of_day.when((hour >= start) & (hour < end)).then(tod_code(start, end))

        time_of_day = time_of_day.when(hour >= end).then(tod_code(end, 24))
        return (
            df.drop_nulls(subset=["timestamp"])
            .unique(subset=["patient_id", "timestamp"], maintain_order=True)
            .select("patient_id", "timestamp", time_of_day.alias("code"))
        )

    return fn


def add_time_derived_measurements_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    INFERRED_STAGE_KEYS = {
        "is_metadata",
        "data_input_dir",
        "metadata_input_dir",
        "output_dir",
        "reducer_output_dir",
    }

    compute_fns = []
    # We use the raw stages object as the induced `stage_cfg` has extra properties like the input and output
    # directories.
    for feature_name, feature_cfg in stage_cfg.items():
        match feature_name:
            case "age":
                compute_fns.append(add_new_events_fntr(age_fntr(feature_cfg)))
            case "time_of_day":
                compute_fns.append(add_new_events_fntr(time_of_day_fntr(feature_cfg)))
            case str() if feature_name in INFERRED_STAGE_KEYS:
                continue
            case _:
                raise ValueError(f"Unknown time-derived measurement: {feature_name}")

        logger.info(f"Adding {feature_name} via config: {OmegaConf.to_yaml(feature_cfg)}")

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        for compute_fn in compute_fns:
            df = compute_fn(df)
        return df

    return fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Adds time-derived measurements to a MEDS cohort as separate observations at each unique timestamp."""

    hydra_loguru_init()
    compute_fn = add_time_derived_measurements_fntr(cfg.stage_cfg)
    map_over(cfg, compute_fn=compute_fn)


if __name__ == "__main__":
    main()
