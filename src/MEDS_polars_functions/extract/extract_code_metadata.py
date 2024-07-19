#!/usr/bin/env python
"""Utilities for extracting code metadata about the codes produced for the MEDS events."""

import copy
import random
from collections.abc import Sequence
from functools import partial, reduce
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig

from MEDS_polars_functions.extract import CONFIG_YAML
from MEDS_polars_functions.mapreduce.mapper import rwlock_wrap
from MEDS_polars_functions.utils import (
    is_col_field,
    parse_col_field,
    stage_init,
    write_lazyframe,
)


def extract_metadata(df: pl.LazyFrame, event_cfg: dict[str, str | None]) -> pl.LazyFrame:
    """Extracts a single metadata dataframe block for an event configuration from the raw metadata.

    Args:
        df: The raw metadata DataFrame. Mandatory columns are determined by the `event_cfg` configuration
            dictionary.
        event_cfg: A dictionary containing the configuration for the event. This must contain the critical
            `"code"` key alongside a mandatory `_metadata` block, which must contain some columns that should
            be extracted from the metadata to link to the code.
            The `"code"` key must contain either (1) a string literal representing the code for the event or
            (2) the name of a column in the raw data from which the code should be extracted. In the latter
            case, the column name should be enclosed in `col()` function call syntax--e.g.,
            `col(my_code_column)`. Note there are no quotes used inside the `col()` function syntax.

    Returns:
        A DataFrame containing the metadata extracted and linked to appropriately constructed code strings for
        the event configuration. The output DataFrame will contain at least two columns: `"code"` and whatever
        metadata column is specified for extraction in the metadata block. The output dataframe will be unique
        by code. Multiple metadata entries for the same code may be collapsed into a list column field in the
        output if the input metadata is not unique by code.

    Raises:
        KeyError: If the event configuration dictionary is missing the `"code"` or `"_metadata"` keys or if
            the `"_metadata_"` key is empty or if columns referenced by the event configuration dictionary are
            not found in the raw metadata.

    Examples:
        >>> extract_metadata(pl.DataFrame(), {})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain 'code' key. Got: []."
        >>> extract_metadata(pl.DataFrame(), {"code": "test"})
        Traceback (most recent call last):
            ...
        KeyError: "Event configuration dictionary must contain '_metadata' key. Got: [code]."
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "B", "C", "D"],
        ...     "code_modifier": ["1", "2", "3", "4"],
        ...     "name": ["Code A-1", "B-2", "C with 3", "D, but 4"],
        ...     "priority": [1, 2, 3, 4],
        ... })
        >>> event_cfg = {
        ...     "code": ["FOO", "col(code)", "col(code_modifier)"],
        ...     "_metadata": {"desc": "name"},
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (4, 2)
        ┌───────────┬──────────┐
        │ code      ┆ desc     │
        │ ---       ┆ ---      │
        │ cat       ┆ str      │
        ╞═══════════╪══════════╡
        │ FOO//A//1 ┆ Code A-1 │
        │ FOO//B//2 ┆ B-2      │
        │ FOO//C//3 ┆ C with 3 │
        │ FOO//D//4 ┆ D, but 4 │
        └───────────┴──────────┘
        >>> raw_metadata = pl.DataFrame({
        ...     "code": ["A", "A", "C", "D"],
        ...     "code_modifier": ["1", "1", "2", "3"],
        ...     "code_modifier_2": ["1", "2", "3", "4"],
        ...     "name": ["A-1-1", "A-1-2", "C-2-3", "D-3-4"],
        ... })
        >>> event_cfg = {
        ...     "code": ["FOO", "col(code)", "col(code_modifier)"],
        ...     "_metadata": {"desc": "name"},
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (4, 2)
        ┌───────────┬────────────────────┐
        │ code      ┆ desc               │
        │ ---       ┆ ---                │
        │ cat       ┆ list[str]          │
        ╞═══════════╪════════════════════╡
        │ FOO//A//1 ┆ [A-1-1,A-1-2]      │
        │ FOO//C//2 ┆ [C-2-3]            │
        │ FOO//D//3 ┆ [D-3-4]            │
        └───────────┴────────────────────┘
    """  # noqa: E501
    df = df
    if "code" not in event_cfg:
        raise KeyError(
            "Event configuration dictionary must contain 'code' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )
    if "_metadata" not in event_cfg:
        raise KeyError(
            "Event configuration dictionary must contain '_metadata' key. "
            f"Got: [{', '.join(event_cfg.keys())}]."
        )

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
    if "numerical_value" in df.columns and not df.schema["numerical_value"].is_numeric():
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

    df = pl.concat(event_dfs, how="diagonal_relaxed")
    return df


def get_events_and_metadata_by_metadata_fp(event_configs):
    raise NotImplementedError("This function is not yet implemented.")


def find_metadata_fp(metadata_input_dir, input_prefix):
    raise NotImplementedError("This function is not yet implemented.")


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Converts the event-sharded raw data into MEDS events and storing them in patient subsharded flat files.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file. It cannot be overwritten
    directly on the command line, but can be overwritten implicitly by overwriting components of the
    `stage_configs.convert_to_sharded_events` key.


    This stage has no stage-specific configuration arguments. It does, naturally, require the global,
    `event_conversion_config_fp` configuration argument to be set to the path of the event conversion yaml
    file.
    """

    stage_input_dir, partial_metadata_dir, _, _ = stage_init(cfg)
    raw_input_dir = Path(cfg.input_dir)

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    partial_metadata_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(event_conversion_cfg, partial_metadata_dir / "event_conversion_config.yaml")

    events_and_metadata_by_metadata_fp = get_events_and_metadata_by_metadata_fp(event_configs)
    event_metadata_configs = list(events_and_metadata_by_metadata_fp.items())
    random.shuffle(event_metadata_configs)

    for input_prefix, event_metadata_cfgs in event_metadata_configs:
        event_metadata_cfgs = copy.deepcopy(event_metadata_cfgs)

        metadata_fp, read_fn = find_metadata_fp(raw_input_dir, input_prefix)
        out_fp = partial_metadata_dir / f"{input_prefix}.parquet"
        logger.info(f"Extracting metadata from {metadata_fp} and saving to {out_fp}")

        compute_fn = partial(extract_metadata, event_cfg=event_metadata_cfgs)

        rwlock_wrap(metadata_fp, out_fp, read_fn, write_lazyframe, compute_fn, do_overwrite=cfg.do_overwrite)

    logger.info("Extracted metadata for all events. Merging.")

    raise NotImplementedError("This function is not yet implemented.")


if __name__ == "__main__":
    main()
