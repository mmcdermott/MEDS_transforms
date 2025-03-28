"""Core utilities for MEDS pipelines built with these tools."""

import logging
from pathlib import Path

import polars as pl
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

from MEDS_transforms import __package_name__, __version__


def get_smallest_valid_uint_type(num: int | float | pl.Expr) -> pl.DataType:
    """Returns the smallest valid unsigned integral type for an ID variable with `num` unique options.

    Args:
        num: The number of IDs that must be uniquely expressed.

    Raises:
        ValueError: If there is no unsigned int type big enough to express the passed number of ID
            variables.

    Examples:
        >>> get_smallest_valid_uint_type(num=1)
        UInt8
        >>> get_smallest_valid_uint_type(num=2**8-1)
        UInt16
        >>> get_smallest_valid_uint_type(num=2**16-1)
        UInt32
        >>> get_smallest_valid_uint_type(num=2**32-1)
        UInt64
        >>> get_smallest_valid_uint_type(num=2**64-1)
        Traceback (most recent call last):
            ...
        ValueError: Value is too large to be expressed as an int!
    """
    if num >= (2**64) - 1:
        raise ValueError("Value is too large to be expressed as an int!")
    if num >= (2**32) - 1:
        return pl.UInt64
    elif num >= (2**16) - 1:
        return pl.UInt32
    elif num >= (2**8) - 1:
        return pl.UInt16
    else:
        return pl.UInt8


def write_lazyframe(df: pl.LazyFrame, out_fp: Path) -> None:
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    df.write_parquet(out_fp, use_pyarrow=True)


def stage_init(cfg: DictConfig) -> tuple[Path, Path, Path]:
    """Initializes the stage by logging the configuration and the stage-specific paths.

    Args:
        cfg: The global configuration object, which should have a ``cfg.stage_cfg`` attribute containing the
            stage specific configuration.

    Returns: The data input directory, stage output directory, and metadata input directory.
    """
    logger.info(f"Running stage with the following configuration:\n{OmegaConf.to_yaml(cfg)}")

    input_dir = Path(cfg.stage_cfg.data_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)
    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)

    def chk(x: Path):
        return "✅" if x.exists() else "❌"

    paths_strs = [
        f"  - {k}: {chk(v)} {str(v.resolve())}"
        for k, v in {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "metadata_input_dir": metadata_input_dir,
        }.items()
    ]

    logger_strs = [
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}",
        "Paths: (checkbox indicates if it exists)",
    ]
    logger.debug("\n".join(logger_strs + paths_strs))

    return input_dir, output_dir, metadata_input_dir


def get_package_name() -> str:
    """Returns the name of the python package running this pipeline.

    Examples:
        >>> get_package_name()
        'MEDS_transforms'
    """
    return __package_name__


def get_package_version() -> str:
    """Returns the version of the python package running this pipeline.

    Examples:
        >>> get_package_version()
        '...'
    """
    return __version__


OmegaConf.register_new_resolver("get_package_version", get_package_version, replace=False)
OmegaConf.register_new_resolver("get_package_name", get_package_name, replace=False)


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
        >>> is_col_field("col(subject_id)")
        True
        >>> is_col_field("col(subject_id")
        False
        >>> is_col_field("subject_id)")
        False
        >>> is_col_field("column(subject_id)")
        False
        >>> is_col_field("subject_id")
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
        >>> parse_col_field("col(subject_id)")
        'subject_id'
        >>> parse_col_field("col(subject_id")
        Traceback (most recent call last):
        ...
        ValueError: Invalid column field: col(subject_id
        >>> parse_col_field("column(subject_id)")
        Traceback (most recent call last):
        ...
        ValueError: Invalid column field: column(subject_id)
    """
    if not is_col_field(field):
        raise ValueError(f"Invalid column field: {field}")
    return field[4:-1]
