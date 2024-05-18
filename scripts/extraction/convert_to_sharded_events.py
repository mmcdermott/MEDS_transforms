#!/usr/bin/env python

import random
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init


def extract_event(df: pl.LazyFrame, event_cfg: dict[str, str | None]) -> pl.LazyFrame:
    """Extracts a single event dataframe from the raw data.

    Args:
        df: The raw data DataFrame. This must have a `"patient_id"` column containing the patient ID. The
            other columns it must have are determined by the `event_cfg` configuration dictionary.
        event_cfg: A dictionary containing the configuration for the event. This must contain two critical
            keys (`"code"` and `"timestamp"`) and may contain additional keys for other columns to include
            in the event DataFrame.
            The `"code"` key must contain either (1) a string literal representing the code for the event or
            (2) the name of a column in the raw data from which the code should be extracted. In the latter
            case, the column name should be enclosed in `col()` function call syntax--e.g.,
            `col(my_code_column)`. Note there are no quotes used inside the `col()` function syntax.
            The `"timestamp"` key must contain either (1) the value `None` if the event has no timestamp
            (e.g., a static event) or (2) the name of a column in the raw data from which the timestamp should
            be extracted. In the latter case, the column name should be enclosed in `col()` function call
            syntax--e.g., `col(my_timestamp_column)`. Note there are no quotes used inside the `col()`
            function syntax.
            If there is a "timestamp_format" key in the `event_cfg` dictionary, the value of this key should
            be a string representing the format of the timestamp column in the raw data. This format should
            conform to the `strftime` format codes. If this key is not present, the timestamp column will be
            parsed as a datetime64 column.
            Any additional key/value pairs in the `event_cfg` dictionary will be interpreted as additional
            columns to extract for the output MEDS data, where the key corresponds to the MEDS column name and
            the value corresponds to the raw name (without any `col()` syntax) of the column in the raw data
            from which the MEDS column should be extracted. These columns must be either numeric or
            categorical (represented as either a `str` or a `Categorical` column in the raw data). Where
            possible, these additional columns should conform to the conventions of the MEDS data schema ---
            e.g., primary numerical values associated with the event should be named `"numerical_value"` in
            the output MEDS data (and thus have the key `"numerical_value"` in the `event_cfg` dictionary).

    Returns:
        A DataFrame containing the event data extracted from the raw data, containing only unique rows across
        all columns. If the raw data has no duplicates when considering the event column space, the output
        dataframe will have the same number of rows as the raw data and be in the same order. The output
        dataframe will contain at least three columns: `"patient_id"`, `"code"`, and `"timestamp"`. If the
        event has additional columns, they will be included in the output dataframe as well.

    Raises:
        KeyError: If the event configuration dictionary is missing the `"code"` or `"timestamp"` keys or if
            columns referenced by the event configuration dictionary are not found in the raw data.

    Examples:
        >>> raw_data = pl.DataFrame({
        ...     "patient_id": [1, 1, 2, 2],
        ...     "code": ["A", "B", "C", "D"],
        ...     "timestamp": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
        ...     "numerical_value": [1, 2, 3, 4],
        ... })
        >>> event_cfg = {
        ...     "code": "col(code)",
        ...     "timestamp": "col(timestamp)",
        ...     "timestamp_format": "%Y-%m-%d",
        ...     "numerical_value": "numerical_value",
        ... }
        >>> extract_event(raw_data, event_cfg)
        shape: (4, 4)
        ┌────────────┬──────┬─────────────────────┬─────────────────┐
        │ patient_id ┆ code ┆ timestamp           ┆ numerical_value │
        │ ---        ┆ ---  ┆ ---                 ┆ ---             │
        │ i64        ┆ cat  ┆ datetime[μs]        ┆ i64             │
        ╞════════════╪══════╪═════════════════════╪═════════════════╡
        │ 1          ┆ A    ┆ 2021-01-01 00:00:00 ┆ 1               │
        │ 1          ┆ B    ┆ 2021-01-02 00:00:00 ┆ 2               │
        │ 2          ┆ C    ┆ 2021-01-03 00:00:00 ┆ 3               │
        │ 2          ┆ D    ┆ 2021-01-04 00:00:00 ┆ 4               │
        └────────────┴──────┴─────────────────────┴─────────────────┘
        >>> from datetime import datetime
        >>> complex_raw_data = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 2, 2, 2, 3],
        ...         "admission_time": [
        ...             "2021-01-01 00:00:00",
        ...             "2021-01-02 00:00:00",
        ...             "2021-01-03 00:00:00",
        ...             "2021-01-04 00:00:00",
        ...             "2021-01-05 00:00:00",
        ...             "2021-01-06 00:00:00",
        ...         ],
        ...         "discharge_time": [
        ...             datetime(2021, 1, 1, 11, 23, 45),
        ...             datetime(2021, 1, 2, 12, 34, 56),
        ...             datetime(2021, 1, 3, 13, 45, 56),
        ...             datetime(2021, 1, 4, 14, 56, 45),
        ...             datetime(2021, 1, 5, 15, 23, 45),
        ...             datetime(2021, 1, 6, 16, 34, 56),
        ...         ],
        ...         "admission_type": ["A", "B", "C", "D", "E", "F"],
        ...         "discharge_location": ["Home", "SNF", "Home", "SNF", "Home", "SNF"],
        ...         "severity_score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ...         "death_time": [
        ...             "2023/01/01",
        ...             "2023/01/01",
        ...             "2023/01/04",
        ...             "2023/01/04",
        ...             "2023/01/04",
        ...             "2023/01/07"
        ...         ],
        ...         "eye_color": ["blue", "blue", "green", "green", "green", "brown"],
        ...     },
        ...     schema={
        ...         "patient_id": pl.UInt8,
        ...         "admission_time": pl.Utf8,
        ...         "discharge_time": pl.Datetime,
        ...         "admission_type": pl.Utf8,
        ...         "discharge_location": pl.Categorical,
        ...         "severity_score": pl.Float64,
        ...         "death_time": pl.Utf8,
        ...         "eye_color": pl.Categorical,
        ...     },
        ... )
        >>> valid_admission_event_cfg = {
        ...     "code": "ADMISSION",
        ...     "timestamp": "col(admission_time)",
        ...     "timestamp_format": "%Y-%m-%d %H:%M:%S",
        ...     "admission_type": "admission_type",
        ...     "severity_on_admission": "severity_score",
        ... }
        >>> valid_discharge_event_cfg = {
        ...     "code": "DISCHARGE",
        ...     "timestamp": "col(discharge_time)",
        ...     "discharge_location": "discharge_location",
        ... }
        >>> valid_death_event_cfg = {
        ...     "code": "DEATH",
        ...     "timestamp": "col(death_time)",
        ...     "timestamp_format": "%Y/%m/%d",
        ... }
        >>> valid_static_event_cfg = {
        ...     "code": "EYE_COLOR",
        ...     "timestamp": None,
        ...     "eye_color": "eye_color",
        ... }
        >>> # We'll print the raw data so you can see what it looks like
        >>> complex_raw_data
        shape: (6, 8)
        ┌────────────┬─────────────────────┬─────────────────────┬────────────────┬────────────────────┬────────────────┬────────────┬───────────┐
        │ patient_id ┆ admission_time      ┆ discharge_time      ┆ admission_type ┆ discharge_location ┆ severity_score ┆ death_time ┆ eye_color │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---            ┆ ---                ┆ ---            ┆ ---        ┆ ---       │
        │ u8         ┆ str                 ┆ datetime[μs]        ┆ str            ┆ cat                ┆ f64            ┆ str        ┆ cat       │
        ╞════════════╪═════════════════════╪═════════════════════╪════════════════╪════════════════════╪════════════════╪════════════╪═══════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 2021-01-01 11:23:45 ┆ A              ┆ Home               ┆ 1.0            ┆ 2023/01/01 ┆ blue      │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 2021-01-02 12:34:56 ┆ B              ┆ SNF                ┆ 2.0            ┆ 2023/01/01 ┆ blue      │
        │ 2          ┆ 2021-01-03 00:00:00 ┆ 2021-01-03 13:45:56 ┆ C              ┆ Home               ┆ 3.0            ┆ 2023/01/04 ┆ green     │
        │ 2          ┆ 2021-01-04 00:00:00 ┆ 2021-01-04 14:56:45 ┆ D              ┆ SNF                ┆ 4.0            ┆ 2023/01/04 ┆ green     │
        │ 2          ┆ 2021-01-05 00:00:00 ┆ 2021-01-05 15:23:45 ┆ E              ┆ Home               ┆ 5.0            ┆ 2023/01/04 ┆ green     │
        │ 3          ┆ 2021-01-06 00:00:00 ┆ 2021-01-06 16:34:56 ┆ F              ┆ SNF                ┆ 6.0            ┆ 2023/01/07 ┆ brown     │
        └────────────┴─────────────────────┴─────────────────────┴────────────────┴────────────────────┴────────────────┴────────────┴───────────┘
        >>> extract_event(complex_raw_data, valid_admission_event_cfg)
        shape: (6, 5)
        ┌────────────┬───────────┬─────────────────────┬────────────────┬───────────────────────┐
        │ patient_id ┆ code      ┆ timestamp           ┆ admission_type ┆ severity_on_admission │
        │ ---        ┆ ---       ┆ ---                 ┆ ---            ┆ ---                   │
        │ u8         ┆ cat       ┆ datetime[μs]        ┆ cat            ┆ f64                   │
        ╞════════════╪═══════════╪═════════════════════╪════════════════╪═══════════════════════╡
        │ 1          ┆ ADMISSION ┆ 2021-01-01 00:00:00 ┆ A              ┆ 1.0                   │
        │ 1          ┆ ADMISSION ┆ 2021-01-02 00:00:00 ┆ B              ┆ 2.0                   │
        │ 2          ┆ ADMISSION ┆ 2021-01-03 00:00:00 ┆ C              ┆ 3.0                   │
        │ 2          ┆ ADMISSION ┆ 2021-01-04 00:00:00 ┆ D              ┆ 4.0                   │
        │ 2          ┆ ADMISSION ┆ 2021-01-05 00:00:00 ┆ E              ┆ 5.0                   │
        │ 3          ┆ ADMISSION ┆ 2021-01-06 00:00:00 ┆ F              ┆ 6.0                   │
        └────────────┴───────────┴─────────────────────┴────────────────┴───────────────────────┘
        >>> extract_event(complex_raw_data, valid_discharge_event_cfg)
        shape: (6, 4)
        ┌────────────┬───────────┬─────────────────────┬────────────────────┐
        │ patient_id ┆ code      ┆ timestamp           ┆ discharge_location │
        │ ---        ┆ ---       ┆ ---                 ┆ ---                │
        │ u8         ┆ cat       ┆ datetime[μs]        ┆ cat                │
        ╞════════════╪═══════════╪═════════════════════╪════════════════════╡
        │ 1          ┆ DISCHARGE ┆ 2021-01-01 11:23:45 ┆ Home               │
        │ 1          ┆ DISCHARGE ┆ 2021-01-02 12:34:56 ┆ SNF                │
        │ 2          ┆ DISCHARGE ┆ 2021-01-03 13:45:56 ┆ Home               │
        │ 2          ┆ DISCHARGE ┆ 2021-01-04 14:56:45 ┆ SNF                │
        │ 2          ┆ DISCHARGE ┆ 2021-01-05 15:23:45 ┆ Home               │
        │ 3          ┆ DISCHARGE ┆ 2021-01-06 16:34:56 ┆ SNF                │
        └────────────┴───────────┴─────────────────────┴────────────────────┘
        >>> extract_event(complex_raw_data, valid_death_event_cfg)
        shape: (3, 3)
        ┌────────────┬───────┬─────────────────────┐
        │ patient_id ┆ code  ┆ timestamp           │
        │ ---        ┆ ---   ┆ ---                 │
        │ u8         ┆ cat   ┆ datetime[μs]        │
        ╞════════════╪═══════╪═════════════════════╡
        │ 1          ┆ DEATH ┆ 2023-01-01 00:00:00 │
        │ 2          ┆ DEATH ┆ 2023-01-04 00:00:00 │
        │ 3          ┆ DEATH ┆ 2023-01-07 00:00:00 │
        └────────────┴───────┴─────────────────────┘
        >>> # Note that the eye color is a static event, so the timestamp is null
        >>> extract_event(complex_raw_data, valid_static_event_cfg)
        shape: (3, 4)
        ┌────────────┬───────────┬──────────────┬───────────┐
        │ patient_id ┆ code      ┆ timestamp    ┆ eye_color │
        │ ---        ┆ ---       ┆ ---          ┆ ---       │
        │ u8         ┆ cat       ┆ datetime[μs] ┆ cat       │
        ╞════════════╪═══════════╪══════════════╪═══════════╡
        │ 1          ┆ EYE_COLOR ┆ null         ┆ blue      │
        │ 2          ┆ EYE_COLOR ┆ null         ┆ green     │
        │ 3          ┆ EYE_COLOR ┆ null         ┆ brown     │
        └────────────┴───────────┴──────────────┴───────────┘
        >>> extract_event(complex_raw_data, {"timestamp": "col(admission_time)"})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' and 'timestamp' keys."
        >>> extract_event(complex_raw_data, {"code": "test"})
        Traceback (most recent call last):
            ..".
        KeyError: "Event configuration dictionary must contain 'code' and 'timestamp' keys."
        >>> extract_event(complex_raw_data, {"code": 34, "timestamp": "col(admission_time)"})
        Traceback (most recent call last):
            ...
        ValueError: Invalid code literal: 34
        >>> extract_event(complex_raw_data, {"code": "test", "timestamp": "12-01-23"})
        Traceback (most recent call last):
            ...
        ValueError: Invalid timestamp literal: 12-01-23
        >>> extract_event(complex_raw_data, {"code": "test", "timestamp": None, "patient_id": 3})
        Traceback (most recent call last):
            ...
        KeyError: "Event column name 'patient_id' cannot be overridden."
        >>> extract_event(complex_raw_data, {"code": "test", "timestamp": None, "foobar": "fuzz"})
        Traceback (most recent call last):
            ...
        KeyError: "Source column 'fuzz' for event column foobar not found in DataFrame schema."
        >>> extract_event(complex_raw_data, {"code": "test", "timestamp": None, "foobar": 32})
        Traceback (most recent call last):
            ...
        ValueError: For event column foobar, source column 32 must be a string column name. Got <class 'int'>.
        >>> extract_event(complex_raw_data, {"code": "test", "timestamp": None, "foobar": "discharge_time"})
        Traceback (most recent call last):
            ...
        ValueError: Source column 'discharge_time' for event column foobar is not numeric or categorical! Cannot be used as an event col.
    """  # noqa: E501
    event_exprs = {"patient_id": pl.col("patient_id")}

    if "code" not in event_cfg or "timestamp" not in event_cfg:
        raise KeyError("Event configuration dictionary must contain 'code' and 'timestamp' keys.")
    if "patient_id" in event_cfg:
        raise KeyError("Event column name 'patient_id' cannot be overridden.")

    code = event_cfg.pop("code")
    match code:
        case str() if code.startswith("col(") and code.endswith(")"):
            event_exprs["code"] = pl.col(code[4:-1]).cast(pl.Categorical)
        case str():
            event_exprs["code"] = pl.lit(code).cast(pl.Categorical)
        case _:
            raise ValueError(f"Invalid code literal: {code}")

    ts = event_cfg.pop("timestamp")
    ts_format = event_cfg.pop("timestamp_format", None)
    match ts:
        case str() if ts.startswith("col(") and ts.endswith(")"):
            if ts_format:
                event_exprs["timestamp"] = pl.col(ts[4:-1]).str.strptime(pl.Datetime, ts_format)
            else:
                event_exprs["timestamp"] = pl.col(ts[4:-1]).cast(pl.Datetime)
        case None:
            event_exprs["timestamp"] = pl.lit(None, dtype=pl.Datetime)
        case _:
            raise ValueError(f"Invalid timestamp literal: {ts}")

    for k, v in event_cfg.items():
        if not isinstance(v, str):
            raise ValueError(
                f"For event column {k}, source column {v} must be a string column name. Got {type(v)}."
            )
        elif v.startswith("col(") and v.endswith(")"):
            logger.warning(
                f"Source column '{v}' for event column {k} is always interpreted as a column name. "
                f"Removing col() function call and setting source column to {v[4:-1]}."
            )
            v = v[4:-1]

        if v not in df.schema:
            raise KeyError(f"Source column '{v}' for event column {k} not found in DataFrame schema.")

        col = pl.col(v)
        if df.schema[v] == pl.Utf8:
            col = col.cast(pl.Categorical)
        elif isinstance(df.schema[v], pl.Categorical):
            pass
        elif not df.schema[v].is_numeric():
            raise ValueError(
                f"Source column '{v}' for event column {k} is not numeric or categorical! "
                "Cannot be used as an event col."
            )

        event_exprs[k] = col
    return df.select(**event_exprs).unique(maintain_order=True)


def convert_to_event(df: pl.LazyFrame, event_cfgs: dict[str, dict[str, str | None]]) -> pl.LazyFrame:
    """Converts a DataFrame of raw data into a DataFrame of events.

    Args:
        df: The raw data DataFrame. This must have a `"patient_id"` column containing the patient ID. The
            other columns it must have are determined by the `event_cfgs` configuration dictionary.
            For the precise mechanism of column determination, see the `extract_event` function.
        event_cfgs: A dictionary containing the configurations for the events to extract. The keys of this
            dictionary should be the names of the events to extract (these are only used for logging, and will
            not automatically appear in any manner in the output data), and the values should be dictionaries
            containing the configuration for each event. Each event configuration dictionary should have the
            same structure as the `event_cfg` dictionary in the `extract_event` function. Please see that
            function for further details on how these configuration dictionaries should be structured and are
            used to extract events.

    Returns:
        A DataFrame containing the events extracted from the raw data. This DataFrame will contain all the
        events extracted from the raw data, with the rows from each event DataFrame concatenated together.
        After concatenation, this dataframe will not be deduplicated, so if the raw data results in duplicates
        across events of different name, these will be preserved in the output DataFrame.
        The output DataFrame will contain at least three columns: `"patient_id"`, `"code"`, and `"timestamp"`.
        If any events have additional columns, these will be included in the output DataFrame as well. All
        columns across all event configurations will be included in the output DataFrame, with `null` values
        filled in for events that do not have a particular column.

    Raises:
        ValueError: If no event configurations are provided or if an error occurs during event extraction.

    Examples:
    """

    if not event_cfgs:
        raise ValueError("No event configurations provided.")

    event_dfs = []
    for event_name, event_cfg in event_cfgs.items():
        try:
            event_dfs.append(extract_event(df, event_cfg))
        except Exception as e:
            raise ValueError(f"Error extracting event {event_name}: {e}") from e

    return pl.concat(event_dfs, how="diagonal")


def filter_and_convert[
    PT_ID_T
](df: pl.LazyFrame, event_cfgs: dict[str, dict[str, str | None]], patients: list[PT_ID_T]) -> pl.LazyFrame:
    """Filters the DataFrame and converts it into events."""

    return convert_to_event(df.filter(pl.col("patient_id").isin(patients)), event_cfgs)


def write_fn(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)


@hydra.main(version_base=None, config_path="configs", config_name="extraction")
def main(cfg: DictConfig):
    """Converts the sub-sharded or raw data into events which are sharded by patient X input shard."""

    hydra_loguru_init()

    raw_cohort_dir = Path(cfg.raw_cohort_dir)
    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)

    shards = MEDS_cohort_dir / "splits.json"

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info(f"Starting event conversion with config:\n{cfg.pretty()}")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)

    patient_subsharded_dir = raw_cohort_dir / "patient_sub_sharded_events"
    OmegaConf.save(event_conversion_cfg, patient_subsharded_dir / "event_conversion_config.yaml")

    patient_splits = list(shards.items())
    random.shuffle(patient_splits)

    event_configs = list(event_conversion_cfg.items())
    random.shuffle(event_configs)

    for sp, patients in patient_splits:
        for input_prefix, event_cfgs in event_configs:
            event_shards = list((raw_cohort_dir / "sub_sharded" / input_prefix).glob("*.parquet"))
            random.shuffle(event_shards)
            for shard_fp in event_shards:
                out_fp = patient_subsharded_dir / sp / input_prefix / shard_fp.name
                logger.info(f"Converting {shard_fp} to events and saving to {out_fp}")

                def compute_fn(df: pl.LazyFrame) -> pl.LazyFrame:
                    try:
                        filter_and_convert(df, event_cfgs=event_cfgs, patients=patients)
                    except Exception as e:
                        raise ValueError(
                            f"Error converting {str(shard_fp.resolve())} for {sp}/{input_prefix}: {e}"
                        ) from e

                rwlock_wrap(shard_fp, out_fp, pl.scan_parquet, write_fn, compute_fn)

    logger.info("Subsharded into converted events.")


if __name__ == "__main__":
    main()
