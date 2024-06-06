"""Utilities for converting input data structures into MEDS events."""

from collections.abc import Sequence
from functools import reduce

import polars as pl
from loguru import logger
from omegaconf.listconfig import ListConfig

from .utils import is_col_field, parse_col_field


def in_format(fmt: str, ts_name: str) -> pl.Expr:
    return pl.col(ts_name).str.strptime(pl.Datetime, fmt, strict=False)


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
        event has additional columns, they will be included in the output dataframe as well. **_Events that
        would be extracted with a null code or a timestamp that should be specified via a column with or
        without a formatting option but in practice is null will be dropped._** Note that this dropping logic
        for code list fields applies to columns in list order -- not to all columns. So, if you have a code
        that starts with a string literal or with a column that is not null, it will always appear, even if a
        subsequent list element is null, just with the missing code columns filled with "UNK".

    Raises:
        KeyError: If the event configuration dictionary is missing the `"code"` or `"timestamp"` keys or if
            columns referenced by the event configuration dictionary are not found in the raw data.

    Examples:
        >>> _ = pl.Config.set_tbl_width_chars(600)
        >>> _ = pl.Config.set_tbl_rows(20)
        >>> _ = pl.Config.set_tbl_cols(20)
        >>> raw_data = pl.DataFrame({
        ...     "patient_id": [1, 1, 2, 2],
        ...     "code": ["A", "B", "C", "D"],
        ...     "code_modifier": ["1", "2", "3", "4"],
        ...     "timestamp": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
        ...     "numerical_value": [1, 2, 3, 4],
        ... })
        >>> event_cfg = {
        ...     "code": ["FOO", "col(code)", "col(code_modifier)"],
        ...     "timestamp": "col(timestamp)",
        ...     "timestamp_format": "%Y-%m-%d",
        ...     "numerical_value": "numerical_value",
        ... }
        >>> extract_event(raw_data, event_cfg)
        shape: (4, 4)
        ┌────────────┬───────────┬─────────────────────┬─────────────────┐
        │ patient_id ┆ code      ┆ timestamp           ┆ numerical_value │
        │ ---        ┆ ---       ┆ ---                 ┆ ---             │
        │ i64        ┆ cat       ┆ datetime[μs]        ┆ i64             │
        ╞════════════╪═══════════╪═════════════════════╪═════════════════╡
        │ 1          ┆ FOO//A//1 ┆ 2021-01-01 00:00:00 ┆ 1               │
        │ 1          ┆ FOO//B//2 ┆ 2021-01-02 00:00:00 ┆ 2               │
        │ 2          ┆ FOO//C//3 ┆ 2021-01-03 00:00:00 ┆ 3               │
        │ 2          ┆ FOO//D//4 ┆ 2021-01-04 00:00:00 ┆ 4               │
        └────────────┴───────────┴─────────────────────┴─────────────────┘
        >>> data_with_nulls = pl.DataFrame({
        ...     "patient_id": [1, 1, 2, 2],
        ...     "code": ["A", None, "C", "D"],
        ...     "code_modifier": ["1", "2", "3", None],
        ...     "timestamp": [None, "2021-01-02", "2021-01-03", "2021-01-04"],
        ...     "numerical_value": [1, 2, 3, 4],
        ... })
        >>> event_cfg = {
        ...     "code": ["col(code)", "col(code_modifier)"],
        ...     "timestamp": "col(timestamp)",
        ...     "timestamp_format": "%Y-%m-%d",
        ...     "numerical_value": "numerical_value",
        ... }
        >>> extract_event(data_with_nulls, event_cfg)
        shape: (2, 4)
        ┌────────────┬────────┬─────────────────────┬─────────────────┐
        │ patient_id ┆ code   ┆ timestamp           ┆ numerical_value │
        │ ---        ┆ ---    ┆ ---                 ┆ ---             │
        │ i64        ┆ cat    ┆ datetime[μs]        ┆ i64             │
        ╞════════════╪════════╪═════════════════════╪═════════════════╡
        │ 2          ┆ C//3   ┆ 2021-01-03 00:00:00 ┆ 3               │
        │ 2          ┆ D//UNK ┆ 2021-01-04 00:00:00 ┆ 4               │
        └────────────┴────────┴─────────────────────┴─────────────────┘
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
        ...         "discharge_status": ["AOx4", "AO", "AAO", "AOx3", "AOx4", "AOx4"],
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
        ...         "discharge_status": pl.Utf8,
        ...         "severity_score": pl.Float64,
        ...         "death_time": pl.Utf8,
        ...         "eye_color": pl.Categorical,
        ...     },
        ... )
        >>> valid_admission_event_cfg = {
        ...     "code": ["ADMISSION", "col(admission_type)"],
        ...     "timestamp": "col(admission_time)",
        ...     "timestamp_format": "%Y-%m-%d %H:%M:%S",
        ...     "numerical_value": "severity_score",
        ... }
        >>> valid_discharge_event_cfg = {
        ...     "code": ["DISCHARGE", "col(discharge_location)"],
        ...     "timestamp": "col(discharge_time)",
        ...     "discharge_status": "discharge_status",
        ... }
        >>> valid_death_event_cfg = {
        ...     "code": "DEATH",
        ...     "timestamp": "col(death_time)",
        ...     "timestamp_format": "%Y/%m/%d",
        ... }
        >>> valid_static_event_cfg = {
        ...     "code": ["EYE_COLOR", "col(eye_color)"],
        ...     "timestamp": None,
        ... }
        >>> # We'll print the raw data so you can see what it looks like
        >>> complex_raw_data
        shape: (6, 9)
        ┌────────────┬─────────────────────┬─────────────────────┬────────────────┬────────────────────┬──────────────────┬────────────────┬────────────┬───────────┐
        │ patient_id ┆ admission_time      ┆ discharge_time      ┆ admission_type ┆ discharge_location ┆ discharge_status ┆ severity_score ┆ death_time ┆ eye_color │
        │ ---        ┆ ---                 ┆ ---                 ┆ ---            ┆ ---                ┆ ---              ┆ ---            ┆ ---        ┆ ---       │
        │ u8         ┆ str                 ┆ datetime[μs]        ┆ str            ┆ cat                ┆ str              ┆ f64            ┆ str        ┆ cat       │
        ╞════════════╪═════════════════════╪═════════════════════╪════════════════╪════════════════════╪══════════════════╪════════════════╪════════════╪═══════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 2021-01-01 11:23:45 ┆ A              ┆ Home               ┆ AOx4             ┆ 1.0            ┆ 2023/01/01 ┆ blue      │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 2021-01-02 12:34:56 ┆ B              ┆ SNF                ┆ AO               ┆ 2.0            ┆ 2023/01/01 ┆ blue      │
        │ 2          ┆ 2021-01-03 00:00:00 ┆ 2021-01-03 13:45:56 ┆ C              ┆ Home               ┆ AAO              ┆ 3.0            ┆ 2023/01/04 ┆ green     │
        │ 2          ┆ 2021-01-04 00:00:00 ┆ 2021-01-04 14:56:45 ┆ D              ┆ SNF                ┆ AOx3             ┆ 4.0            ┆ 2023/01/04 ┆ green     │
        │ 2          ┆ 2021-01-05 00:00:00 ┆ 2021-01-05 15:23:45 ┆ E              ┆ Home               ┆ AOx4             ┆ 5.0            ┆ 2023/01/04 ┆ green     │
        │ 3          ┆ 2021-01-06 00:00:00 ┆ 2021-01-06 16:34:56 ┆ F              ┆ SNF                ┆ AOx4             ┆ 6.0            ┆ 2023/01/07 ┆ brown     │
        └────────────┴─────────────────────┴─────────────────────┴────────────────┴────────────────────┴──────────────────┴────────────────┴────────────┴───────────┘
        >>> extract_event(complex_raw_data, valid_admission_event_cfg)
        shape: (6, 4)
        ┌────────────┬──────────────┬─────────────────────┬─────────────────┐
        │ patient_id ┆ code         ┆ timestamp           ┆ numerical_value │
        │ ---        ┆ ---          ┆ ---                 ┆ ---             │
        │ u8         ┆ cat          ┆ datetime[μs]        ┆ f64             │
        ╞════════════╪══════════════╪═════════════════════╪═════════════════╡
        │ 1          ┆ ADMISSION//A ┆ 2021-01-01 00:00:00 ┆ 1.0             │
        │ 1          ┆ ADMISSION//B ┆ 2021-01-02 00:00:00 ┆ 2.0             │
        │ 2          ┆ ADMISSION//C ┆ 2021-01-03 00:00:00 ┆ 3.0             │
        │ 2          ┆ ADMISSION//D ┆ 2021-01-04 00:00:00 ┆ 4.0             │
        │ 2          ┆ ADMISSION//E ┆ 2021-01-05 00:00:00 ┆ 5.0             │
        │ 3          ┆ ADMISSION//F ┆ 2021-01-06 00:00:00 ┆ 6.0             │
        └────────────┴──────────────┴─────────────────────┴─────────────────┘
        >>> extract_event(complex_raw_data, valid_discharge_event_cfg)
        shape: (6, 4)
        ┌────────────┬─────────────────┬─────────────────────┬──────────────────┐
        │ patient_id ┆ code            ┆ timestamp           ┆ discharge_status │
        │ ---        ┆ ---             ┆ ---                 ┆ ---              │
        │ u8         ┆ cat             ┆ datetime[μs]        ┆ cat              │
        ╞════════════╪═════════════════╪═════════════════════╪══════════════════╡
        │ 1          ┆ DISCHARGE//Home ┆ 2021-01-01 11:23:45 ┆ AOx4             │
        │ 1          ┆ DISCHARGE//SNF  ┆ 2021-01-02 12:34:56 ┆ AO               │
        │ 2          ┆ DISCHARGE//Home ┆ 2021-01-03 13:45:56 ┆ AAO              │
        │ 2          ┆ DISCHARGE//SNF  ┆ 2021-01-04 14:56:45 ┆ AOx3             │
        │ 2          ┆ DISCHARGE//Home ┆ 2021-01-05 15:23:45 ┆ AOx4             │
        │ 3          ┆ DISCHARGE//SNF  ┆ 2021-01-06 16:34:56 ┆ AOx4             │
        └────────────┴─────────────────┴─────────────────────┴──────────────────┘
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
        shape: (3, 3)
        ┌────────────┬──────────────────┬──────────────┐
        │ patient_id ┆ code             ┆ timestamp    │
        │ ---        ┆ ---              ┆ ---          │
        │ u8         ┆ cat              ┆ datetime[μs] │
        ╞════════════╪══════════════════╪══════════════╡
        │ 1          ┆ EYE_COLOR//blue  ┆ null         │
        │ 2          ┆ EYE_COLOR//green ┆ null         │
        │ 3          ┆ EYE_COLOR//brown ┆ null         │
        └────────────┴──────────────────┴──────────────┘
        >>> extract_event(complex_raw_data, {"timestamp": "col(admission_time)"})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' key. Got: [timestamp]."
        >>> extract_event(complex_raw_data, {"code": "test", "value": "severity_score"})
        Traceback (most recent call last):
            ..".
        KeyError: "Event configuration dictionary must contain 'timestamp' key. Got: [code, value]."
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

    if "code" not in event_cfg:
        raise KeyError(
            "Event configuration dictionary must contain 'code' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )
    if "timestamp" not in event_cfg:
        raise KeyError(
            "Event configuration dictionary must contain 'timestamp' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )
    if "patient_id" in event_cfg:
        raise KeyError("Event column name 'patient_id' cannot be overridden.")

    codes = event_cfg.pop("code")
    if not isinstance(codes, (list, ListConfig)):
        logger.debug(
            f"Event code '{codes}' is a {type(codes)}, not a list. Automatically converting to a list."
        )
        codes = [codes]

    code_exprs = []
    code_null_filter_expr = None
    for i, code in enumerate(codes):
        match code:
            case str() if is_col_field(code) and parse_col_field(code) in df.schema:
                code_col = parse_col_field(code)
                logger.info(f"Extracting code column {code_col}")
                code_exprs.append(pl.col(code_col).cast(pl.Utf8).fill_null("UNK"))
                if i == 0:
                    code_null_filter_expr = pl.col(code_col).is_not_null()
            case str():
                logger.info(f"Adding code literate {code}")
                code_exprs.append(pl.lit(code, dtype=pl.Utf8))
            case _:
                raise ValueError(f"Invalid code literal: {code}")
    event_exprs["code"] = reduce(lambda a, b: a + pl.lit("//") + b, code_exprs).cast(pl.Categorical)

    ts = event_cfg.pop("timestamp")
    ts_format = event_cfg.pop("timestamp_format", None)
    if isinstance(ts_format, str):
        ts_format = [ts_format]

    ts_filter_expr = None
    match ts:
        case str() if is_col_field(ts):
            ts_name = parse_col_field(ts)
            if isinstance(ts_format, (ListConfig, list)):
                logger.info(f"Adding timestamp column {ts_name} in possible formats {', '.join(ts_format)}")
                assert len(ts_format) > 0, "Timestamp format list is empty"
                event_exprs["timestamp"] = pl.coalesce(*(in_format(fmt, ts_name) for fmt in ts_format))
            else:
                logger.info(f"{ts_name} should already be in Date/time format")
                assert ts_format is None
                event_exprs["timestamp"] = pl.col(ts_name).cast(pl.Datetime)
            ts_filter_expr = event_exprs["timestamp"].is_not_null()
        case None:
            logger.info("Adding null literate for timestamp")
            event_exprs["timestamp"] = pl.lit(None, dtype=pl.Datetime)
        case _:
            raise ValueError(f"Invalid timestamp literal: {ts}")

    for k, v in event_cfg.items():
        if not isinstance(v, str):
            raise ValueError(
                f"For event column {k}, source column {v} must be a string column name. Got {type(v)}."
            )
        elif is_col_field(v):
            logger.warning(
                f"Source column '{v}' for event column {k} is always interpreted as a column name. "
                f"Removing col() function call and setting source column to {parse_col_field(v)}."
            )
            v = parse_col_field(v)

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

    if code_null_filter_expr is not None:
        logger.info(f"Filtering out rows with null codes via {code_null_filter_expr}")
        df = df.filter(code_null_filter_expr)
    if ts_filter_expr is not None:
        logger.info(f"Filtering out rows with null timestamps via {ts_filter_expr}")
        df = df.filter(ts_filter_expr)

    df = df.select(**event_exprs).unique(maintain_order=True)

    # if numerical_value column is not numeric, convert it to float
    if "numerical_value" in df.columns:
        if not df.schema["numerical_value"].is_numeric():
            logger.warning(f"Converting numerical_value to float for codes {codes}")
        df = df.with_columns(pl.col("numerical_value").cast(pl.Float64, strict=False))

    return df


def convert_to_events(
    df: pl.LazyFrame, event_cfgs: dict[str, dict[str, str | None | Sequence[str]]]
) -> pl.LazyFrame:
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
        >>> _ = pl.Config.set_tbl_width_chars(600)
        >>> _ = pl.Config.set_tbl_rows(20)
        >>> _ = pl.Config.set_tbl_cols(20)
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
        >>> event_cfgs = {
        ...     "admission": {
        ...         "code": "ADMISSION",
        ...         "timestamp": "col(admission_time)",
        ...         "timestamp_format": "%Y-%m-%d %H:%M:%S",
        ...         "admission_type": "admission_type",
        ...         "severity_on_admission": "severity_score",
        ...     },
        ...     "discharge": {
        ...         "code": "DISCHARGE",
        ...         "timestamp": "col(discharge_time)",
        ...         "discharge_location": "discharge_location",
        ...     },
        ...     "death": {
        ...         "code": "DEATH",
        ...         "timestamp": "col(death_time)",
        ...         "timestamp_format": "%Y/%m/%d",
        ...     },
        ...     "eye_color": {
        ...         "code": "EYE_COLOR",
        ...         "timestamp": None,
        ...         "eye_color": "eye_color",
        ...     },
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
        >>> convert_to_events(complex_raw_data, event_cfgs)
        shape: (18, 7)
        ┌────────────┬───────────┬─────────────────────┬────────────────┬───────────────────────┬────────────────────┬───────────┐
        │ patient_id ┆ code      ┆ timestamp           ┆ admission_type ┆ severity_on_admission ┆ discharge_location ┆ eye_color │
        │ ---        ┆ ---       ┆ ---                 ┆ ---            ┆ ---                   ┆ ---                ┆ ---       │
        │ u8         ┆ cat       ┆ datetime[μs]        ┆ cat            ┆ f64                   ┆ cat                ┆ cat       │
        ╞════════════╪═══════════╪═════════════════════╪════════════════╪═══════════════════════╪════════════════════╪═══════════╡
        │ 1          ┆ ADMISSION ┆ 2021-01-01 00:00:00 ┆ A              ┆ 1.0                   ┆ null               ┆ null      │
        │ 1          ┆ ADMISSION ┆ 2021-01-02 00:00:00 ┆ B              ┆ 2.0                   ┆ null               ┆ null      │
        │ 2          ┆ ADMISSION ┆ 2021-01-03 00:00:00 ┆ C              ┆ 3.0                   ┆ null               ┆ null      │
        │ 2          ┆ ADMISSION ┆ 2021-01-04 00:00:00 ┆ D              ┆ 4.0                   ┆ null               ┆ null      │
        │ 2          ┆ ADMISSION ┆ 2021-01-05 00:00:00 ┆ E              ┆ 5.0                   ┆ null               ┆ null      │
        │ 3          ┆ ADMISSION ┆ 2021-01-06 00:00:00 ┆ F              ┆ 6.0                   ┆ null               ┆ null      │
        │ 1          ┆ DISCHARGE ┆ 2021-01-01 11:23:45 ┆ null           ┆ null                  ┆ Home               ┆ null      │
        │ 1          ┆ DISCHARGE ┆ 2021-01-02 12:34:56 ┆ null           ┆ null                  ┆ SNF                ┆ null      │
        │ 2          ┆ DISCHARGE ┆ 2021-01-03 13:45:56 ┆ null           ┆ null                  ┆ Home               ┆ null      │
        │ 2          ┆ DISCHARGE ┆ 2021-01-04 14:56:45 ┆ null           ┆ null                  ┆ SNF                ┆ null      │
        │ 2          ┆ DISCHARGE ┆ 2021-01-05 15:23:45 ┆ null           ┆ null                  ┆ Home               ┆ null      │
        │ 3          ┆ DISCHARGE ┆ 2021-01-06 16:34:56 ┆ null           ┆ null                  ┆ SNF                ┆ null      │
        │ 1          ┆ DEATH     ┆ 2023-01-01 00:00:00 ┆ null           ┆ null                  ┆ null               ┆ null      │
        │ 2          ┆ DEATH     ┆ 2023-01-04 00:00:00 ┆ null           ┆ null                  ┆ null               ┆ null      │
        │ 3          ┆ DEATH     ┆ 2023-01-07 00:00:00 ┆ null           ┆ null                  ┆ null               ┆ null      │
        │ 1          ┆ EYE_COLOR ┆ null                ┆ null           ┆ null                  ┆ null               ┆ blue      │
        │ 2          ┆ EYE_COLOR ┆ null                ┆ null           ┆ null                  ┆ null               ┆ green     │
        │ 3          ┆ EYE_COLOR ┆ null                ┆ null           ┆ null                  ┆ null               ┆ brown     │
        └────────────┴───────────┴─────────────────────┴────────────────┴───────────────────────┴────────────────────┴───────────┘
        >>> convert_to_events(complex_raw_data, {})
        Traceback (most recent call last):
            ...
        ValueError: No event configurations provided.
        >>> convert_to_events(complex_raw_data, {"admission": {}})
        Traceback (most recent call last):
            ...
        ValueError: Error extracting event admission: ...
    """  # noqa: E501

    if not event_cfgs:
        raise ValueError("No event configurations provided.")

    event_dfs = []
    for event_name, event_cfg in event_cfgs.items():
        try:
            logger.info(f"Building computational graph for extracting {event_name}")
            event_dfs.append(extract_event(df, event_cfg))
        except Exception as e:
            raise ValueError(f"Error extracting event {event_name}: {e}") from e

    df = pl.concat(event_dfs, how="diagonal")
    return df
