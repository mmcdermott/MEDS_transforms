"""A functor to compute timeline token measurements on a MEDS dataset."""

import logging
from collections.abc import Callable

import polars as pl
from meds import DataSchema
from omegaconf import DictConfig

from ...compute_modes.compute_fn import identity_fn
from .utils import normalize_time_unit, unique_events

logger = logging.getLogger(__name__)


def timeline_tokens_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Create a function that adds timeline-measurement rows between events (distinct timepoints).

    In this context, "timeline-measurements" correspond to added measurements that capture properties about
    the timeline of the subject's events -- namely, the start of their longitudinal (non-static) data, the
    time between subsequent events in their data, and the end of their longitudinal data.

    Args:
        cfg: The configuration for the time delta function. This must contain the following keys:
            - "time_delta_code": The code for time-delta measurements in the output data. If not provided,
              defaults to `"TIMELINE//DELTA//{time_unit}"`. If the code string provided contains
              `{time_unit}`, it will be substituted for the specified time unit (in normalized form). If
              `null`, these rows will not be added.
            - "time_unit": The unit for the time deltas when dates are converted to a numeric value in the
              output data. This is mandatory of time delta codes are added at all. For the full list of units
              supported, see `utils.normalize_time_unit`.
            - "timeline_start_code": The code for the "start of temporal data" code in the output data. If
              `null`, these rows will not be added. If not provided, defaults to `"TIMELINE//START"`.
            - "timeline_end_code": The code for the "end of temporal data" code in the output data. If `null`,
              these rows will not be added. If not provided, defaults to `"TIMELINE//END"`.

    > [!NOTE]
    > If a subject has only static data, no time delta or timeline start/end tokens will be added.

    Returns:
        A function that returns the to-be-added "time_delta" measurements between subsequent events in a
        subject's data. The very first event for a subject has a null time_delta, so it is imputed with a
        special time start token.

    Raises:
        ValueError: If the input unit is not recognized.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 1, 1, 2, 2, 3, 3],
        ...         "time": [
        ...             None,
        ...             datetime(1990, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(1988, 1, 2),
        ...             datetime(2023, 1, 3),
        ...             None,
        ...             None,
        ...         ],
        ...         "code": [
        ...             "static", "MEDS_BIRTH", "lab//A", "lab//B", "rx", "MEDS_BIRTH", "lab//A", "foo", "bar"
        ...         ],
        ...     },
        ...     schema={"subject_id": pl.UInt32, "time": pl.Datetime, "code": pl.Utf8},
        ... )
        >>> df
        shape: (9, 3)
        ┌────────────┬─────────────────────┬────────────┐
        │ subject_id ┆ time                ┆ code       │
        │ ---        ┆ ---                 ┆ ---        │
        │ u32        ┆ datetime[μs]        ┆ str        │
        ╞════════════╪═════════════════════╪════════════╡
        │ 1          ┆ null                ┆ static     │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ MEDS_BIRTH │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A     │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B     │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ rx         │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ MEDS_BIRTH │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A     │
        │ 3          ┆ null                ┆ foo        │
        │ 3          ┆ null                ┆ bar        │
        └────────────┴─────────────────────┴────────────┘

    If we simply provide a time unit, we get start, end, and delta tokens in the specified unit. Note the
    output column name depends on the normalized time unit.

        >>> timeline_tokens_fn = timeline_tokens_fntr(DictConfig({"time_unit": "day"}))
        >>> timeline_tokens_fn(df)
        shape: (7, 4)
        ┌────────────┬─────────────────────┬───────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code                  ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                   ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str                   ┆ f32           │
        ╞════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
        │ 1          ┆ 1990-01-01 00:00:00 ┆ TIMELINE//START       ┆ null          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ TIMELINE//DELTA//days ┆ 11323.0       │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ TIMELINE//DELTA//days ┆ 1.0           │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ TIMELINE//END         ┆ null          │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ TIMELINE//START       ┆ null          │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ TIMELINE//DELTA//days ┆ 12785.0       │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ TIMELINE//END         ┆ null          │
        └────────────┴─────────────────────┴───────────────────────┴───────────────┘

    We can change the codes for any of the timeline token types and the unit used for the time deltas via the
    config:

        >>> cfg = DictConfig({
        ...     "time_unit": "y",
        ...     "timeline_start_code": "ST",
        ...     "timeline_end_code": "END",
        ...     "time_delta_code": "DELTA//{time_unit}",
        ... })
        >>> timeline_tokens_fn = timeline_tokens_fntr(cfg)
        >>> timeline_tokens_fn(df)
        shape: (7, 4)
        ┌────────────┬─────────────────────┬──────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code         ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---          ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str          ┆ f32           │
        ╞════════════╪═════════════════════╪══════════════╪═══════════════╡
        │ 1          ┆ 1990-01-01 00:00:00 ┆ ST           ┆ null          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ DELTA//years ┆ 31.001347     │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ DELTA//years ┆ 0.002738      │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ END          ┆ null          │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ ST           ┆ null          │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ DELTA//years ┆ 35.004169     │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ END          ┆ null          │
        └────────────┴─────────────────────┴──────────────┴───────────────┘

    We can also set the time code for deltas to not contain the unit string:

        >>> cfg = DictConfig({
        ...     "time_unit": "mos", # Months
        ...     "time_delta_code": "DELTA",
        ... })
        >>> timeline_tokens_fn = timeline_tokens_fntr(cfg)
        >>> timeline_tokens_fn(df)
        shape: (7, 4)
        ┌────────────┬─────────────────────┬─────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code            ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---             ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str             ┆ f32           │
        ╞════════════╪═════════════════════╪═════════════════╪═══════════════╡
        │ 1          ┆ 1990-01-01 00:00:00 ┆ TIMELINE//START ┆ null          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ DELTA           ┆ 372.015839    │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ DELTA           ┆ 0.032855      │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ TIMELINE//END   ┆ null          │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ TIMELINE//START ┆ null          │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ DELTA           ┆ 420.049683    │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ TIMELINE//END   ┆ null          │
        └────────────┴─────────────────────┴─────────────────┴───────────────┘

    We can disable any or all of the components by setting their code field to `null`.

        >>> timeline_tokens_fn = timeline_tokens_fntr(DictConfig({"time_delta_code": None}))
        >>> timeline_tokens_fn(df)
        shape: (4, 3)
        ┌────────────┬─────────────────────┬─────────────────┐
        │ subject_id ┆ time                ┆ code            │
        │ ---        ┆ ---                 ┆ ---             │
        │ u32        ┆ datetime[μs]        ┆ str             │
        ╞════════════╪═════════════════════╪═════════════════╡
        │ 1          ┆ 1990-01-01 00:00:00 ┆ TIMELINE//START │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ TIMELINE//END   │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ TIMELINE//START │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ TIMELINE//END   │
        └────────────┴─────────────────────┴─────────────────┘
        >>> cfg = DictConfig({
        ...     "time_unit": "year",
        ...     "timeline_start_code": None,
        ...     "timeline_end_code": None,
        ... })
        >>> timeline_tokens_fn = timeline_tokens_fntr(cfg)
        >>> timeline_tokens_fn(df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬────────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code                   ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                    ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str                    ┆ f32           │
        ╞════════════╪═════════════════════╪════════════════════════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ TIMELINE//DELTA//years ┆ 31.001347     │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ TIMELINE//DELTA//years ┆ 0.002738      │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ TIMELINE//DELTA//years ┆ 35.004169     │
        └────────────┴─────────────────────┴────────────────────────┴───────────────┘

    If all are None, the function will return an identity function that does not modify the input
    argument.

        >>> cfg = DictConfig({
        ...     "timeline_start_code": None,
        ...     "timeline_end_code": None,
        ...     "time_delta_code": None,
        ... })
        >>> timeline_tokens_fn = timeline_tokens_fntr(cfg)
        >>> timeline_tokens_fn(df)
        shape: (9, 3)
        ┌────────────┬─────────────────────┬────────────┐
        │ subject_id ┆ time                ┆ code       │
        │ ---        ┆ ---                 ┆ ---        │
        │ u32        ┆ datetime[μs]        ┆ str        │
        ╞════════════╪═════════════════════╪════════════╡
        │ 1          ┆ null                ┆ static     │
        │ 1          ┆ 1990-01-01 00:00:00 ┆ MEDS_BIRTH │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A     │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//B     │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ rx         │
        │ 2          ┆ 1988-01-02 00:00:00 ┆ MEDS_BIRTH │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ lab//A     │
        │ 3          ┆ null                ┆ foo        │
        │ 3          ┆ null                ┆ bar        │
        └────────────┴─────────────────────┴────────────┘
        >>> timeline_tokens_fn("foobar") # This won't raise an error as this is truly an identity function.
        'foobar'
    """

    timeline_start_token = cfg.get("timeline_start_code", "TIMELINE//START")
    do_add_timeline_start = timeline_start_token is not None

    timeline_end_token = cfg.get("timeline_end_code", "TIMELINE//END")
    do_add_timeline_end = timeline_end_token is not None

    time_delta_token = cfg.get("time_delta_code", "TIMELINE//DELTA//{time_unit}")
    do_add_time_delta = time_delta_token is not None

    if do_add_time_delta:
        normalized_unit, seconds_in_unit = normalize_time_unit(cfg.time_unit)
        microseconds_in_unit = int(1e6) * seconds_in_unit
        time_delta_token = time_delta_token.format(time_unit=normalized_unit)

    if not (do_add_timeline_start or do_add_time_delta or do_add_timeline_end):
        logger.warning(
            "No timeline tokens were configured to be added. This is likely an issue with your config. "
            "Returning an identity function."
        )
        return identity_fn

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        events = unique_events(df)

        out_measurements = []

        time = pl.col(DataSchema.time_name)
        code_dtype = df.collect_schema().get(DataSchema.code_name, pl.Utf8)
        numeric_value_dtype = df.collect_schema().get(DataSchema.numeric_value_name, pl.Float32)

        if do_add_timeline_start:
            first_time = time.min().over(DataSchema.subject_id_name)
            timeline_start_code = pl.lit(timeline_start_token, dtype=code_dtype)

            out_measurements.append(
                events.filter(pl.col(DataSchema.time_name) == first_time).with_columns(
                    timeline_start_code.alias(DataSchema.code_name)
                )
            )

        if do_add_time_delta:
            time_delta_code = pl.lit(time_delta_token, dtype=code_dtype)

            time_delta = time.diff(null_behavior="ignore").over(DataSchema.subject_id_name)
            timeline_deltas_expr = (time_delta.dt.total_microseconds() / microseconds_in_unit).cast(
                numeric_value_dtype
            )

            out_measurements.append(
                events.with_columns(
                    time_delta_code.alias(DataSchema.code_name),
                    timeline_deltas_expr.alias(DataSchema.numeric_value_name),
                ).filter(pl.col(DataSchema.numeric_value_name).is_not_null())
            )

        if do_add_timeline_end:
            last_time = time.max().over(DataSchema.subject_id_name)
            timeline_end_code = pl.lit(timeline_end_token, dtype=code_dtype)

            out_measurements.append(
                events.filter(pl.col(DataSchema.time_name) == last_time).with_columns(
                    timeline_end_code.alias(DataSchema.code_name)
                )
            )

        return pl.concat(out_measurements, how="diagonal").sort(
            by=[DataSchema.subject_id_name, DataSchema.time_name], maintain_order=True
        )

    return fn
