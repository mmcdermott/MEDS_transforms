"""A functor to compute age measurements on a MEDS dataset."""

import logging
from collections.abc import Callable

import polars as pl
from meds import DataSchema
from omegaconf import DictConfig

from .utils import normalize_time_unit, unique_events

logger = logging.getLogger(__name__)


def age_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Create a function that adds a subject's age to a DataFrame.

    Args:
        cfg: The configuration for the age function. This must contain the following mandatory keys:
            - "DOB_code": The code for the date of birth event in the raw data. A string literal. Defaults to
              None. This is used preferentially over the "DOB_regex" key if both are provided.
            - "DOB_regex": A regex pattern to match the date of birth event in the raw data. Defaults to
              "^MEDS_BIRTH(//.*)?"
            - "age_code": The code for the age event in the output data. Defaults to "AGE".
            - "age_unit": The unit for the age event when converted to a numeric value in the output data.
              Defaults to "years"

    Returns:
        A function that returns the to-be-added "age" events with the subject's age for all input events with
        unique, non-null times in the data, for all subjects who have an observed date of birth. It does
        not add an event for times that are equal to the date of birth.

    Raises:
        ValueError: If the input unit is not recognized.

    Examples:
        >>> from datetime import datetime
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
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 1),
        ...         ],
        ...         "code": ["static", "DOB", "lab//A", "lab//B", "rx", "DOB", "lab//A", "lab//B", "dx//1"],
        ...     },
        ...     schema={"subject_id": pl.UInt32, "time": pl.Datetime, "code": pl.Utf8},
        ... )
        >>> df
        shape: (9, 3)
        ┌────────────┬─────────────────────┬────────┐
        │ subject_id ┆ time                ┆ code   │
        │ ---        ┆ ---                 ┆ ---    │
        │ u32        ┆ datetime[μs]        ┆ str    │
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
        >>> age_fn = age_fntr(DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "years"}))
        >>> age_fn(df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str  ┆ f32           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE  ┆ 31.001347     │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ AGE  ┆ 31.004084     │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE  ┆ 35.004169     │
        └────────────┴─────────────────────┴──────┴───────────────┘

    You can also use regular expressions for DOB codes. The default is "^MEDS_BIRTH(//.*)?", which matches
    optional prefixes in case the birth events have code modifiers:

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
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 1),
        ...         ],
        ...         "code": [
        ...             "static",
        ...             "MEDS_BIRTH",
        ...             "lab//A",
        ...             "lab//B",
        ...             "rx",
        ...             "MEDS_BIRTH//home",
        ...             "lab//A",
        ...             "lab//B",
        ...             "dx//1",
        ...         ],
        ...     },
        ...     schema={"subject_id": pl.UInt32, "time": pl.Datetime, "code": pl.Utf8},
        ... )
        >>> cfg = DictConfig({"DOB_regex": "^MEDS_BIRTH(//.*)?", "age_code": "AGE", "age_unit": "days"})
        >>> age_fn = age_fntr(cfg)
        >>> age_fn(df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ str  ┆ f32           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ AGE  ┆ 11323.0       │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ AGE  ┆ 11324.0       │
        │ 2          ┆ 2023-01-03 00:00:00 ┆ AGE  ┆ 12785.0       │
        └────────────┴─────────────────────┴──────┴───────────────┘

    Errors are thrown if invalid units are provided or if the configuration doesn't have one of either
    `DOB_code` or `DOB_regex`:

        >>> age_fntr(DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "scores"}))
        Traceback (most recent call last):
            ...
        ValueError: Unknown time unit 'scores'. Valid units include:
        ...
        >>> age_fntr(DictConfig({"age_code": "AGE", "age_unit": "y"}))
        Traceback (most recent call last):
            ...
        ValueError: Either 'DOB_regex' or 'DOB_code' must be provided in the configuration.
    """

    _, seconds_in_unit = normalize_time_unit(cfg.get("age_unit", "y"))
    microseconds_in_unit = int(1e6) * seconds_in_unit

    if cfg.get("DOB_code", None) is not None:
        is_dob = pl.col(DataSchema.code_name).str.contains(cfg.DOB_code, literal=True)
    elif cfg.get("DOB_regex", None) is not None:
        is_dob = pl.col(DataSchema.code_name).str.contains(cfg.DOB_regex)
    else:
        raise ValueError("Either 'DOB_regex' or 'DOB_code' must be provided in the configuration.")

    dob_col = "__dob"

    age_expr = (pl.col(DataSchema.time_name) - pl.col(dob_col)).dt.total_microseconds() / microseconds_in_unit
    age_expr = age_expr.cast(pl.Float32, strict=False)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        code_dtype = df.collect_schema().get(DataSchema.code_name, pl.Utf8)

        dobs = df.filter(is_dob).select(
            DataSchema.subject_id_name, pl.col(DataSchema.time_name).alias(dob_col)
        )
        events = unique_events(df)

        return (
            events.join(dobs, on=DataSchema.subject_id_name, how="inner", maintain_order="left")
            .select(
                DataSchema.subject_id_name,
                DataSchema.time_name,
                pl.lit(cfg.get("age_code", "AGE"), dtype=code_dtype).alias(DataSchema.code_name),
                age_expr.alias(DataSchema.numeric_value_name),
            )
            .drop_nulls(subset=[DataSchema.numeric_value_name])
            .filter(pl.col(DataSchema.numeric_value_name) > 0)
        )

    return fn
