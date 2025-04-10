"""Transformations for extracting numeric and/or categorical values from the MEDS dataset."""

from collections.abc import Callable
import logging

from meds import code_field, numeric_value_field, subject_id_field, time_field
from omegaconf import DictConfig
import polars as pl

from ... import INFERRED_STAGE_KEYS
from ...parser import cfg_to_expr
from .. import Stage

logger = logging.getLogger(__name__)

MANDATORY_TYPES = {
    subject_id_field: pl.Int64,
    time_field: pl.Datetime("us"),
    code_field: pl.String,
    numeric_value_field: pl.Float32,
    "categorical_value": pl.String,
    "text_value": pl.String,
}


@Stage.register
def extract_values(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Create a function that extracts values from a MEDS cohort.

    This functor does not filter the applied dataframe prior to applying the extraction process. It is likely
    best used with match & revise to filter the data before applying the extraction process.

    Args:
        stage_cfg: The configuration for the extraction stage. This should be a mapping from output column
            name to the parser configuration for the value you want to extract from the MEDS measurement for
            that column.

    Returns:
        A function that takes a LazyFrame and returns a LazyFrame with the original data and the extracted
        values.

    Examples:
        >>> stage_cfg = {"numeric_value": "foo", "categorical_value": "bar"}
        >>> fn = extract_values(stage_cfg)
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1],
        ...         "time": [1, 2, 3],
        ...         "foo": ["1", "2", "3"],
        ...         "bar": [1.0, 2.0, 4.0],
        ...     }
        ... )
        >>> fn(df)
        shape: (3, 6)
        ┌────────────┬──────┬─────┬─────┬───────────────┬───────────────────┐
        │ subject_id ┆ time ┆ foo ┆ bar ┆ numeric_value ┆ categorical_value │
        │ ---        ┆ ---  ┆ --- ┆ --- ┆ ---           ┆ ---               │
        │ i64        ┆ i64  ┆ str ┆ f64 ┆ f32           ┆ str               │
        ╞════════════╪══════╪═════╪═════╪═══════════════╪═══════════════════╡
        │ 1          ┆ 1    ┆ 1   ┆ 1.0 ┆ 1.0           ┆ 1.0               │
        │ 1          ┆ 2    ┆ 2   ┆ 2.0 ┆ 2.0           ┆ 2.0               │
        │ 1          ┆ 3    ┆ 3   ┆ 4.0 ┆ 3.0           ┆ 4.0               │
        └────────────┴──────┴─────┴─────┴───────────────┴───────────────────┘
        >>> stage_cfg = {32: "foo"}
        >>> fn = extract_values(stage_cfg)
        Traceback (most recent call last):
            ...
        ValueError: Invalid column name: 32
        >>> stage_cfg = {"numeric_value": {"lit": 1}}
        >>> fn = extract_values(stage_cfg)
        Traceback (most recent call last):
            ...
        ValueError: Error building expression for numeric_value...
        >>> stage_cfg = {"numeric_value": "foo", "categorical_value": "bar"}
        >>> fn = extract_values(stage_cfg)
        >>> df = pl.DataFrame({"subject_id": [1, 1, 1], "time": [1, 2, 3]})
        >>> fn(df)
        Traceback (most recent call last):
            ...
        ValueError: Missing columns: ['bar', 'foo']

    Note that deprecated column names like "numerical_value" or "timestamp" won't be re-typed.

        >>> stage_cfg = {"numerical_value": "foo"}
        >>> fn = extract_values(stage_cfg)
        >>> df = pl.DataFrame({"subject_id": [1, 1, 1], "time": [1, 2, 3], "foo": ["1", "2", "3"]})
        >>> fn(df)
        shape: (3, 4)
        ┌────────────┬──────┬─────┬─────────────────┐
        │ subject_id ┆ time ┆ foo ┆ numerical_value │
        │ ---        ┆ ---  ┆ --- ┆ ---             │
        │ i64        ┆ i64  ┆ str ┆ str             │
        ╞════════════╪══════╪═════╪═════════════════╡
        │ 1          ┆ 1    ┆ 1   ┆ 1               │
        │ 1          ┆ 2    ┆ 2   ┆ 2               │
        │ 1          ┆ 3    ┆ 3   ┆ 3               │
        └────────────┴──────┴─────┴─────────────────┘

    If we try to extract a column that is in the MEDS mandatory fields, we will get a warning.

        >>> stage_cfg = {"subject_id": "foo"}
        >>> with print_warnings():
        ...     fn = extract_values(stage_cfg)
        Warning: You should almost CERTAINLY not be extracting subject_id as a value.
    """

    new_cols = []
    need_cols = set()
    for out_col_n, value_cfg in stage_cfg.items():
        if out_col_n in INFERRED_STAGE_KEYS:
            continue

        try:
            expr, cols = cfg_to_expr(value_cfg)
        except ValueError as e:
            raise ValueError(f"Error building expression for {out_col_n}") from e

        match out_col_n:
            case str() if out_col_n in MANDATORY_TYPES:
                expr = expr.cast(MANDATORY_TYPES[out_col_n])
                if out_col_n in (subject_id_field, time_field):
                    logger.warning(f"You should almost CERTAINLY not be extracting {out_col_n} as a value.")
            case str():
                pass
            case _:
                raise ValueError(f"Invalid column name: {out_col_n}")

        new_cols.append(expr.alias(out_col_n))
        need_cols.update(cols)

    def map_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        in_cols = set(df.collect_schema().names())
        if not need_cols.issubset(in_cols):
            raise ValueError(f"Missing columns: {sorted(list(need_cols - in_cols))}")

        return df.with_columns(new_cols).sort(subject_id_field, "time", maintain_order=True)

    return map_fn
