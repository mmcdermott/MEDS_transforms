"""A polars-to-polars transformation function for filtering patients by sequence length."""

from collections.abc import Callable

import polars as pl
from omegaconf import DictConfig


def filter_codes_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Filters patient events to only encompass those with a set of permissible codes.

    Args:
        df: The input DataFrame.
        stage_cfg: The configuration for the code filtering stage.

    Returns:
        The processed DataFrame.

    Examples:
        >>> raise NotImplementedError
    """

    raise NotImplementedError


def filter_outliers_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Filters patient events to only encompass those with a set of permissible codes.

    Args:
        df: The input DataFrame.
        stage_cfg: The configuration for the code filtering stage.

    Returns:
        The processed DataFrame.

    Examples:
        >>> raise NotImplementedError
    """

    raise NotImplementedError
