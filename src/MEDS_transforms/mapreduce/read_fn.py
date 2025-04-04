"""Functionality for reading input data shards."""

import logging
from collections.abc import Callable
from pathlib import Path

import polars as pl

from .types import DF_T

logger = logging.getLogger(__name__)

READ_FN_T = Callable[[Path], DF_T]


def read_and_filter_fntr(filter_expr: pl.Expr, read_fn: READ_FN_T) -> READ_FN_T:
    """Create a function that reads a DataFrame from a file and filters it based on a given expression.

    This is specified as a functor in this way to allow it to modify arbitrary other read functions for use in
    different mapreduce pipelines.

    Args:
        filter_expr: The filter expression to apply to the DataFrame.
        read_fn: The read function to use to read the DataFrame.

    Returns:
        A function that reads a DataFrame from a file and filters it based on the given expression.

    Examples:
        >>> dfs = {
        ...     "df1": pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
        ...     "df2": pl.DataFrame({"a": [4, 5, 6], "b": [7, 8, 9]})
        ... }
        >>> read_fn = lambda key: dfs[key]
        >>> fn = read_and_filter_fntr((pl.col("a") % 2) == 0, read_fn)
        >>> fn("df1")
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 5   │
        └─────┴─────┘
        >>> fn("df2")
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 4   ┆ 7   │
        │ 6   ┆ 9   │
        └─────┴─────┘
        >>> fn = read_and_filter_fntr((pl.col("b") % 2) == 0, read_fn)
        >>> fn("df1")
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> fn("df2")
        shape: (1, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 5   ┆ 8   │
        └─────┴─────┘
    """

    def read_and_filter(in_fp: Path) -> DF_T:
        return read_fn(in_fp).filter(filter_expr)

    return read_and_filter
