#!/usr/bin/env python
"""Utilities for extracting code metadata about the codes produced for the MEDS events."""

import copy
import random
from functools import partial, reduce
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig

from MEDS_polars_functions.extract import CONFIG_YAML
from MEDS_polars_functions.extract.utils import get_supported_fp
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
        metadata column is specified for extraction in the metadata block. The output dataframe will not
        necessarily be unique by code if the input metadata is not unique by code.

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
        ...     "title": ["A-1-1", "A-1-2", "C-2-3", "D-3-4"],
        ... })
        >>> event_cfg = {
        ...     "code": ["FOO", "col(code)", "col(code_modifier)"],
        ...     "_metadata": {
        ...         "desc": "title",
        ...         "parent_code": {
        ...             "OUT_VAL_1": {"code_modifier_2": 2},
        ...             "OUT_VAL_2": {"code_modifier_2": 3},
        ...         },
        ...     },
        ... }
        >>> extract_metadata(raw_metadata, event_cfg)
        shape: (4, 2)
        ┌───────────┬───────┬───────────────┐
        │ code      ┆ desc  ┆ parent_code   │
        │ ---       ┆ ---   ┆ ---           │
        │ cat       ┆ str   ┆ str           │
        ╞═══════════╪═══════╪═══════════════╡
        │ FOO//A//1 ┆ A-1-1 ┆               │
        │ FOO//A//1 ┆ A-1-2 ┆ OUT_VAL_1     │
        │ FOO//C//2 ┆ C-2-3 ┆ OUT_VAL_2     │
        │ FOO//D//3 ┆ D-3-4 ┆               │
        └───────────┴───────┴───────────────┘
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


def get_events_and_metadata_by_metadata_fp(event_configs):
    """This reformats the event conversion config to map metadata file input prefixes to linked event
    configs."""

    raise NotImplementedError("This function is not yet implemented.")


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """TODO."""

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

    # Load all codes
    all_codes = (
        pl.scan_parquet(stage_input_dir / "**/*.parquet")
        .select(pl.col("code").unique())
        .collect()
        .get_column("code")
        .to_list()
    )

    all_out_fps = []
    for input_prefix, event_metadata_cfgs in event_metadata_configs:
        event_metadata_cfgs = copy.deepcopy(event_metadata_cfgs)

        metadata_fp, read_fn = get_supported_fp(raw_input_dir, input_prefix)
        out_fp = partial_metadata_dir / f"{input_prefix}.parquet"
        logger.info(f"Extracting metadata from {metadata_fp} and saving to {out_fp}")

        compute_fn = partial(extract_metadata, event_cfg=event_metadata_cfgs, allowed_codes=all_codes)

        rwlock_wrap(metadata_fp, out_fp, read_fn, write_lazyframe, compute_fn, do_overwrite=cfg.do_overwrite)
        all_out_fps.append(out_fp)

    logger.info("Extracted metadata for all events. Merging.")

    if cfg.worker != 0:
        logger.info("Code metadata extraction completed. Exiting")
        return

    logger.info("Starting reduction process")

    while not all(fp.is_file() for fp in all_out_fps):
        logger.info("Waiting to begin reduction for all files to be written...")
        time.sleep(cfg.polling_time)

    start = datetime.now()
    logger.info("All map shards complete! Starting code metadata reduction computation.")
    reducer_fn = partial(pl.concat, how="vertical")

    reduced = reducer_fn(*[pl.scan_parquet(fp, glob=False) for fp in all_out_fps])
    join_cols = ["code", *code_modifier_cols]
    metadata_cols = [c for c in reduced.columns if c not in join_cols]
    reduced = reduced.group_by(join_cols).agg(*(pl.col(c) for c in metadata_cols)).collect()

    reducer_fp = Path(cfg.cohort_dir) / "code_metadata.parquet"

    if reducer_fp.exists():
        logger.info(f"Joining to existing code metadata at {str(reducer_fp.resolve())}")
        existing = pl.read_parquet(reducer_fp, use_pyarrow=True)
        reduced = existing.join(reduced, on=join_cols, how="outer")

    pl.write_parquet(reduced, reducer_fp, use_pyarrow=True)
    logger.info(f"Finished reduction in {datetime.now() - start}")


if __name__ == "__main__":
    main()
