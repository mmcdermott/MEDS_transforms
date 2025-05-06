"""Utilities to help add time-derived measurements (e.g., a subject's age) to a MEDS dataset."""

import logging

import polars as pl
from meds import DataSchema

logger = logging.getLogger(__name__)

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


def unique_events(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Returns rows corresponding to unique events (time-points) in a MEDS dataframe.

    Args:
        df: The input DataFrame.

    Returns:
        A DataFrame containing rows with unique time-points for each subject, with only subject ID and time
        columns included.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 1, 2, 2],
        ...         "time": [
        ...             None,
        ...             datetime(2021, 1, 1, 12, 0),
        ...             datetime(2021, 1, 1, 12, 0),
        ...             datetime(2022, 1, 1, 18, 0),
        ...             datetime(2022, 1, 1, 18, 0),
        ...             datetime(2022, 1, 1, 18, 0),
        ...         ],
        ...         "code": ["static", "lab//A", "lab//A", "lab//B", "lab//B", "dx//1"],
        ...     },
        ... )
        >>> df
        shape: (6, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ subject_id ┆ time                ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ i64        ┆ datetime[μs]        ┆ str    │
        ╞════════════╪═════════════════════╪════════╡
        │ 1          ┆ null                ┆ static │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ lab//A │
        │ 1          ┆ 2021-01-01 12:00:00 ┆ lab//A │
        │ 1          ┆ 2022-01-01 18:00:00 ┆ lab//B │
        │ 2          ┆ 2022-01-01 18:00:00 ┆ lab//B │
        │ 2          ┆ 2022-01-01 18:00:00 ┆ dx//1  │
        └────────────┴─────────────────────┴────────┘
        >>> unique_events(df)
        shape: (3, 2)
        ┌────────────┬─────────────────────┐
        │ subject_id ┆ time                │
        │ ---        ┆ ---                 │
        │ i64        ┆ datetime[μs]        │
        ╞════════════╪═════════════════════╡
        │ 1          ┆ 2021-01-01 12:00:00 │
        │ 1          ┆ 2022-01-01 18:00:00 │
        │ 2          ┆ 2022-01-01 18:00:00 │
        └────────────┴─────────────────────┘
    """

    return (
        df.drop_nulls(subset=[DataSchema.time_name])
        .select(DataSchema.subject_id_name, DataSchema.time_name)
        .unique(maintain_order=True)
    )
