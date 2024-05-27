"""Core utilities for MEDS pipelines built with these tools."""

import os
import sys
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import OmegaConf


def current_script_name() -> str:
    """Returns the name of the script that called this function.

    Returns:
        str: The name of the script that called this function.
    """
    return Path(sys.argv[0]).stem


def populate_stage(
    stage_name: str,
    input_dir: str,
    cohort_dir: str,
    stages: list[str],
    stage_configs: dict[str, dict],
    pre_parsed_stages: dict[str, dict] | None = None,
) -> dict:
    """Populates a stage in the stages configuration with inferred stage parameters.

    Infers and adds (unless already present, in which case the provided value is used) the following
    parameters to the stage configuration:
      - `is_metadata`: Whether the stage is a metadata stage, which is determined to be `False` if the stage
        does not have an `aggregations` parameter.
      - `data_input_dir`: The input directory for the stage (either the global input directory or the previous
        data stage's output directory).
      - `metadata_input_dir`: The input directory for the stage (either the global input directory or the
        previous metadata stage's output directory).
      - `output_dir`: The output directory for the stage (the cohort directory with the stage name appended).

    Args:
        stage_name: The name of the stage to populate.
        input_dir: The global input directory.
        cohort_dir: The cohort directory into which this overall pipeline is writing data.
        stages: The names of the stages processed by this pipeline in order.
        stage_configs: The raw, unresolved stage configuration dictionaries for any stages with specific
            arguments, keyed by stage name.
        pre_parsed_stages: The stages configuration dictionaries (resolved), keyed by stage name. If
            specified, the function will not re-resolve the stages in this list.

    Returns:
        dict: The populated stage configuration.

    Raises:
        ValueError: If the stage is not present in the stages configuration.

    Examples:
        >>> from omegaconf import DictConfig
        >>> root_config = DictConfig({
        ...     "input_dir": "/a/b",
        ...     "cohort_dir": "/c/d",
        ...     "stages": ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"],
        ...     "stage_configs": {
        ...         "stage2": {"is_metadata": True},
        ...         "stage3": {"is_metadata": None},
        ...         "stage4": {"data_input_dir": "/e/f", "output_dir": "/g/h"},
        ...         "stage5": {"aggregations": ["foo"]},
        ...     },
        ... })
        >>> args = [root_config[k] for k in ["input_dir", "cohort_dir", "stages", "stage_configs"]]
        >>> populate_stage("stage1", *args) # doctest: +NORMALIZE_WHITESPACE
        {'is_metadata': False, 'data_input_dir': '/a/b', 'metadata_input_dir': '/a/b',
         'output_dir': '/c/d/stage1'}
        >>> populate_stage("stage2", *args) # doctest: +NORMALIZE_WHITESPACE
        {'is_metadata': True, 'data_input_dir': '/c/d/stage1', 'metadata_input_dir': '/a/b',
         'output_dir': '/c/d/stage2'}
        >>> populate_stage("stage3", *args) # doctest: +NORMALIZE_WHITESPACE
        {'is_metadata': False, 'data_input_dir': '/c/d/stage1',
         'metadata_input_dir': '/c/d/stage2', 'output_dir': '/c/d/stage3'}
        >>> populate_stage("stage4", *args) # doctest: +NORMALIZE_WHITESPACE
        {'data_input_dir': '/e/f', 'output_dir': '/g/h', 'is_metadata': False,
         'metadata_input_dir': '/c/d/stage2'}
        >>> populate_stage("stage5", *args) # doctest: +NORMALIZE_WHITESPACE
        {'aggregations': ['foo'], 'is_metadata': True, 'data_input_dir': '/g/h',
         'metadata_input_dir': '/c/d/stage2', 'output_dir': '/c/d/stage5'}
        >>> populate_stage("stage6", *args) # doctest: +NORMALIZE_WHITESPACE
        {'is_metadata': False, 'data_input_dir': '/g/h',
         'metadata_input_dir': '/c/d/stage5', 'output_dir': '/c/d/stage6'}
        >>> populate_stage("stage7", *args) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: 'stage7' is not a valid stage name. Options are: stage1, stage2, stage3, stage4, stage5,
            stage6
    """

    for s in stage_configs.keys():
        if s not in stages:
            raise ValueError(
                f"stage config key '{s}' is not a valid stage name. Options are: {list(stages.keys())}"
            )

    if stage_name not in stages:
        raise ValueError(f"'{stage_name}' is not a valid stage name. Options are: {', '.join(stages)}")

    if pre_parsed_stages is None:
        pre_parsed_stages = {}

    stage = None
    prior_data_stage = None
    prior_metadata_stage = None
    for s in stages:
        if s == stage_name:
            stage = stage_configs.get(s, {})
            break
        elif s in pre_parsed_stages:
            s_resolved = pre_parsed_stages[s]
        else:
            s_resolved = populate_stage(s, input_dir, cohort_dir, stages, stage_configs, pre_parsed_stages)

        pre_parsed_stages[s] = s_resolved
        if s_resolved["is_metadata"]:
            prior_metadata_stage = s_resolved
        else:
            prior_data_stage = s_resolved

    logger.debug(
        f"Parsing stage {stage_name}:\nResolved prior data stage: {prior_data_stage}\n"
        f"Resolved prior metadata stage: {prior_metadata_stage}"
    )

    inferred_keys = {
        "is_metadata": "aggregations" in stage,
        "data_input_dir": input_dir if prior_data_stage is None else prior_data_stage["output_dir"],
        "metadata_input_dir": (
            input_dir if prior_metadata_stage is None else prior_metadata_stage["output_dir"]
        ),
        "output_dir": os.path.join(cohort_dir, stage_name),
    }

    out = {**stage}
    for key, val in inferred_keys.items():
        if key not in out or out[key] is None:
            out[key] = val

    return out


OmegaConf.register_new_resolver("current_script_name", current_script_name, replace=False)
OmegaConf.register_new_resolver("populate_stage", populate_stage, replace=False)


def hydra_loguru_init() -> None:
    """Adds loguru output to the logs that hydra scrapes.

    Must be called from a hydra main!
    """
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.add(os.path.join(hydra_path, "main.log"))


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
