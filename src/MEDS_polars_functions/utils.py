"""Core utilities for MEDS pipelines built with these tools."""

import os
from pathlib import Path

from omegaconf import OmegaConf
import hydra
import polars as pl
from loguru import logger as log

def get_stage_input_dir(
    raw_input_dir: str, cohort_dir: str, stages: list[str], stage: str
) -> str:
    """Resolves the input directory for a stage in a MEDS pipeline.

    Args:
        raw_input_dir: The raw input directory (used as the input when the stage is the 1st stage).
        cohort_dir: The cohort (output) directory; used as the source for the default stage output.
        stages: The stages in the pipeline.
        stage: The current stage.

    Returns:
        The input directory for the current stage.

    Examples:
        >>> get_stage_input_dir("/a/b", "/c/d", ["stage1", "stage2"], "stage1")
        '/a/b'
        >>> get_stage_input_dir("/a/b", "/c/d", ["stage1", "stage2"], "stage2")
        '/c/d/stage1'
    """
    if stage == stages[0]:
        return raw_input_dir
    elif stage not in stages:
        raise ValueError(
            f"Can't impute input directory for {stage} as it is not in the stages list! "
            f"Stages: {stages}. "
            "If this is intentional, please provide the input directory explicitly or remove the "
            "attempted interpolation from your config by overwriting the `stage_input_dir` parameter."
        )
    return os.path.join(cohort_dir, stages[stages.index(stage) - 1])

# We actually call this here that way it is registered in every script when the module is imported.
OmegaConf.register_new_resolver("stage_input_idr", get_stage_input_dir, replace=True)

def hydra_loguru_init() -> None:
    """Adds loguru output to the logs that hydra scrapes.

    Must be called from a hydra main!
    """
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.add(os.path.join(hydra_path, "main.log"))


def write_lazyframe(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)


def get_shard_prefix(base_path: Path, fp: Path) -> str:
    """Extracts the shard prefix from a file path by removing the raw_cohort_dir.

    Args:
        base_path: The base path to remove.
        fp: The file path to extract the shard prefix from.

    Returns:
        The shard prefix (the file path relative to the base path with the suffix removed).

    Examples:
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d.parquet"))
        'd'
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d/e.csv.gz"))
        'd/e'
    """

    relative_path = fp.relative_to(base_path)
    relative_parent = relative_path.parent
    file_name = relative_path.name.split(".")[0]

    return str(relative_parent / file_name)


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
