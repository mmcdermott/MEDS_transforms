"""Core utilities for MEDS pipelines built with these tools."""

import os
from pathlib import Path

import hydra
import polars as pl
from loguru import logger as log


def hydra_loguru_init() -> None:
    """Adds loguru output to the logs that hydra scrapes.

    Must be called from a hydra main!
    """
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.add(os.path.join(hydra_path, "main.log"))


def write_lazyframe(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)


def is_col_field(field: str | None) -> bool:
    """Checks if a string field is formatted as "col(column_name)".

    This format is used to denote a column in a Polars DataFrame in the event conversion configuration.

    Args:
        field (str | None): The field to check.

    Returns:
        bool: True if the field is formatted as "col(column_name)", False otherwise.

    Examples:
        >>> is_col_field("col(patient_id)")
        True
        >>> is_col_field("col(patient_id")
        False
        >>> is_col_field("patient_id)")
        False
        >>> is_col_field("column(patient_id)")
        False
        >>> is_col_field("patient_id")
        False
        >>> is_col_field(None)
        False
    """
    if field is None:
        return False
    return field.startswith("col(") and field.endswith(")")


def parse_col_field(field: str) -> str:
    """Extracts the actual column name from a string formatted as "col(column_name)".

    Args:
        field (str): A string formatted as "col(column_name)".

    Raises:
        ValueError: If the input string does not match the expected format.

    Examples:
        >>> parse_col_field("col(patient_id)")
        'patient_id'
        >>> parse_col_field("col(patient_id")
        Traceback (most recent call last):
        ...
        ValueError: Invalid column field: col(patient_id
        >>> parse_col_field("column(patient_id)")
        Traceback (most recent call last):
        ...
        ValueError: Invalid column field: column(patient_id)
    """
    if not is_col_field(field):
        raise ValueError(f"Invalid column field: {field}")
    return field[4:-1]
