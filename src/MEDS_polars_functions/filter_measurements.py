"""A polars-to-polars transformation function for filtering patients by sequence length."""

from collections.abc import Callable

import polars as pl
from omegaconf import DictConfig


def filter_codes_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifier_columns: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Filters patient events to only encompass those with a set of permissible codes.

    Args:
        df: The input DataFrame.
        stage_cfg: The configuration for the code filtering stage.

    Returns:
        The processed DataFrame.

    Examples:
        >>> code_metadata_df = pl.DataFrame({
        ...     "code": pl.Series(["A", "A", "B", "C"], dtype=pl.Categorical),
        ...     "modifier1": [1, 2, 1, 2],
        ...     "code/n_patients":  [1, 1, 2, 2],
        ...     "code/n_occurrences": [2, 1, 3, 2],
        ... })
        >>> raise NotImplementedError
    """

    raise NotImplementedError


def filter_outliers_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifier_columns: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Filters patient events to only encompass those with a set of permissible codes.

    Args:
        df: The input DataFrame.
        stage_cfg: The configuration for the code filtering stage.

    Returns:
        The processed DataFrame.

    Examples:
        >>> code_metadata_df = pl.DataFrame({
        ...     "code": pl.Series(["A", "A", "B", "C"], dtype=pl.Categorical),
        ...     "modifier1": [1, 2, 1, 2],
        ...     "code/n_patients":  [1, 1, 2, 2],
        ...     "code/n_occurrences": [2, 1, 3, 2],
        ...     "values/n_patients":  [1, 1, 2, 2],
        ...     "values/n_occurrences": [2, 1, 3, 2],
        ...     "values/n_ints": [0, 1, 3, 1],
        ...     "values/sum": [2.2, 6.0, 14.0, 12.5],
        ...     "values/sum_sqd": [2.42, 36.0, 84.0, 81.25],
        ...     "values/min": [0, -1, 2, 2.],
        ...     "values/max": [1.1, 6.0, 8.0, 7.5],
        ... })
        >>> raise NotImplementedError
    """

    raise NotImplementedError
