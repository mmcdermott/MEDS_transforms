#!/usr/bin/env python
"""A polars-to-polars transformation function for filtering patients by sequence length."""
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from MEDS_polars_functions.mapreduce.mapper import map_over
from MEDS_polars_functions.utils import get_smallest_valid_uint
from omegaconf import DictConfig, ListConfig

pl.enable_string_cache()


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
        >>> code_metadata_df = pl.DataFrame({
        ...     "code":       pl.Series(["A",  "A",  "B",  "C"], dtype=pl.Categorical),
        ...     "modifier1":            [1,    2,    1,    2],
        ...     "values/n_occurrences": [3,    1,    3,    2],
        ...     "values/sum":           [0.0,  4.0,  12.0, 2.0],
        ...     "values/sum_sqd":       [27.0, 16.0, 75.0, 4.0],
        ... # for clarity: ----- mean = [0.0,  4.0,  4.0,  1.0]
        ... # for clarity: --- stddev = [3.0,  0.0,  3.0,  1.0]
        ... })
        >>> data = pl.DataFrame({
        ...     "patient_id":      [1,   1,   2,   2],
        ...     "code":  pl.Series(["A", "B", "A", "C"], dtype=pl.Categorical),
        ...     "modifier1":       [1,   1,   2,   2],
        ... # for clarity: mean    [0.0, 4.0, 4.0, 1.0]
        ... # for clarity: stddev  [3.0, 3.0, 0.0, 1.0]
        ...     "numerical_value": [15., 16., 3.9, 1.0],
        ... }).lazy()
        >>> stage_cfg = DictConfig({})
        >>> fn = reorder_by_code_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
    """

    ordered_code_patterns = stage_cfg.get("ordered_code_patterns", None)
    if not ordered_code_patterns:
        return lambda df: df

    if not isinstance(ordered_code_patterns, (list, ListConfig)):
        raise ValueError("TODO")

    num_code_patterns = len(ordered_code_patterns)
    code_pattern_idx_dtype = get_smallest_valid_uint(num_code_patterns + 1)  # TODO: make function

    join_cols = ["code"]
    if code_modifier_columns:
        raise NotImplementedError("This transformation doesn't currently support code modifier columns.")
        join_cols.extend(code_modifier_columns)

    cols_to_select = ["code"]
    if code_modifier_columns:
        cols_to_select.extend(code_modifier_columns)

    code = pl.col("code").cast(pl.Utf8)

    code_order_idx_exprs = pl  # this let's us chain the when/then/otherwise calls equivalently over the loop.
    for i, code_matcher in enumerate(ordered_code_patterns):
        matcher_expr = code.str.contains(code_matcher)
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
            .sort("patient_id", "timestamp", "code_order_idx", maintain_order=True)
            .drop("code_order_idx")
        )

    return reorder_fn


config_yaml = files("MEDS_polars_functions").joinpath("configs/preprocess.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """TODO: Put script help string here"""

    code_metadata = pl.read_parquet(
        Path(cfg.stage_cfg.metadata_input_dir) / "code_metadata.parquet", use_pyarrow=True
    )

    map_over(cfg, compute_fn=reorder_by_code_fntr(cfg.stage_cfg, code_metadata))


if __name__ == "__main__":
    main()
