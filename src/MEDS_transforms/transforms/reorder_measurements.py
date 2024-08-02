#!/usr/bin/env python
"""A polars-to-polars transformation function for filtering patients by sequence length."""
from collections.abc import Callable
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, ListConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.utils import get_smallest_valid_uint_type


def reorder_by_code_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifier_columns: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Re-orders a dataframe within the temporal and subject ID ordering via a specified code order.

    Args:
        stage_cfg: TODO
        code_metadata: TODO
        code_modifier_columns: TODO

    Returns:
        A function with signature `Callable[[pl.LazyFrame], pl.LazyFrame]` that re-orders the input DataFrame.

    Examples:
        >>> code_metadata_df = pl.DataFrame({"code": ["A", "A", "B", "C"], "modifier1": [1, 2, 1, 2]})
        >>> data = pl.DataFrame({
        ...     "patient_id":[1, 1, 2, 2], "time": [1, 1, 1, 1],
        ...     "code": ["A", "B", "A", "C"], "modifier1": [1, 2, 1, 2]
        ... })
        >>> stage_cfg = DictConfig({"ordered_code_patterns": ["B", "A"]})
        >>> fn = reorder_by_code_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data.lazy()).collect()
        shape: (4, 4)
        ┌────────────┬──────┬──────┬───────────┐
        │ patient_id ┆ time ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---  ┆ ---       │
        │ i64        ┆ i64  ┆ str  ┆ i64       │
        ╞════════════╪══════╪══════╪═══════════╡
        │ 1          ┆ 1    ┆ B    ┆ 2         │
        │ 1          ┆ 1    ┆ A    ┆ 1         │
        │ 2          ┆ 1    ┆ A    ┆ 1         │
        │ 2          ┆ 1    ┆ C    ┆ 2         │
        └────────────┴──────┴──────┴───────────┘

    TODO: A more complex test with true regexes.
    """

    ordered_code_patterns = stage_cfg.get("ordered_code_patterns", None)
    if not ordered_code_patterns:
        return lambda df: df

    if not isinstance(ordered_code_patterns, (list, ListConfig)):
        raise ValueError(
            f"The 'ordered_code_patterns' field must be a list of strings. Got {ordered_code_patterns}."
        )
    for pattern in ordered_code_patterns:
        if not isinstance(pattern, str):
            raise ValueError(f"Each element of 'ordered_code_patterns' must be a string. Got {pattern}.")

    num_code_patterns = len(ordered_code_patterns)
    code_pattern_idx_dtype = get_smallest_valid_uint_type(num_code_patterns + 1)  # TODO: make function

    join_cols = ["code"]
    if code_modifier_columns:
        logger.warning("Code reordering currently only matches against the 'code' column, not code modifiers")
        join_cols.extend(code_modifier_columns)

    cols_to_select = ["code"]
    if code_modifier_columns:
        cols_to_select.extend(code_modifier_columns)

    code_order_idx_exprs = pl  # this let's us chain the when/then/otherwise calls equivalently over the loop.
    for i, code_matcher in enumerate(ordered_code_patterns):
        matcher_expr = pl.col("code").str.contains(code_matcher)
        idx_expr = pl.lit(i, dtype=code_pattern_idx_dtype)
        code_order_idx_exprs = code_order_idx_exprs.when(matcher_expr).then(idx_expr)

    code_order_idx_exprs = code_order_idx_exprs.otherwise(num_code_patterns).alias("code_order_idx")

    code_indices = code_metadata.lazy().select(*join_cols, code_order_idx_exprs)

    def reorder_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Reorders.

        TODO
        """

        return (
            df.join(code_indices, on=join_cols, how="left", coalesce=True)
            .sort("patient_id", "time", "code_order_idx", maintain_order=True)
            .drop("code_order_idx")
        )

    return reorder_fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO: Put script help string here"""

    code_metadata = pl.read_parquet(
        Path(cfg.stage_cfg.metadata_input_dir) / "codes.parquet", use_pyarrow=True
    )

    map_over(cfg, compute_fn=reorder_by_code_fntr(cfg.stage_cfg, code_metadata))


if __name__ == "__main__":
    main()
