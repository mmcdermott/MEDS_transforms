#!/usr/bin/env python
"""A polars-to-polars transformation function for filtering patients by sequence length."""
from collections.abc import Callable
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_polars_functions import PREPROCESS_CONFIG_YAML
from MEDS_polars_functions.mapreduce.mapper import map_over

pl.enable_string_cache()


def occlude_outliers_fntr(
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
        >>> stage_cfg = DictConfig({"stddev_cutoff": 4.5})
        >>> fn = occlude_outliers_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (4, 5)
        ┌────────────┬──────┬───────────┬─────────────────┬───────────────────────────┐
        │ patient_id ┆ code ┆ modifier1 ┆ numerical_value ┆ numerical_value/is_inlier │
        │ ---        ┆ ---  ┆ ---       ┆ ---             ┆ ---                       │
        │ i64        ┆ cat  ┆ i64       ┆ f64             ┆ bool                      │
        ╞════════════╪══════╪═══════════╪═════════════════╪═══════════════════════════╡
        │ 1          ┆ A    ┆ 1         ┆ null            ┆ false                     │
        │ 1          ┆ B    ┆ 1         ┆ 16.0            ┆ true                      │
        │ 2          ┆ A    ┆ 2         ┆ null            ┆ false                     │
        │ 2          ┆ C    ┆ 2         ┆ 1.0             ┆ true                      │
        └────────────┴──────┴───────────┴─────────────────┴───────────────────────────┘
    """

    stddev_cutoff = stage_cfg.get("stddev_cutoff", None)
    if stddev_cutoff is None:
        return lambda df: df

    join_cols = ["code"]
    if code_modifier_columns:
        join_cols.extend(code_modifier_columns)

    cols_to_select = ["code"]
    if code_modifier_columns:
        cols_to_select.extend(code_modifier_columns)

    mean_col = pl.col("values/sum") / pl.col("values/n_occurrences")
    stddev_col = (pl.col("values/sum_sqd") / pl.col("values/n_occurrences") - mean_col**2) ** 0.5
    if "values/mean" not in code_metadata.columns:
        cols_to_select.append(mean_col.alias("values/mean"))
    if "values/std" not in code_metadata.columns:
        cols_to_select.append(stddev_col.alias("values/std"))

    code_metadata = code_metadata.lazy().select(cols_to_select)

    def occlude_outliers_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        f"""Filters out outlier numerical values from patient events.

        In particular, this function filters the DataFrame to only include numerical values that are within
        {stddev_cutoff} standard deviations of the mean for the corresponding (code, modifier) pair.
        """

        val = pl.col("numerical_value")
        mean = pl.col("values/mean")
        stddev = pl.col("values/std")
        filter_expr = (val - mean).abs() <= stddev_cutoff * stddev

        return (
            df.join(code_metadata, on=join_cols, how="left", coalesce=True)
            .with_columns(
                filter_expr.alias("numerical_value/is_inlier"),
                pl.when(filter_expr).then(pl.col("numerical_value")).alias("numerical_value"),
            )
            .drop("values/mean", "values/std")
        )

    return occlude_outliers_fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    code_metadata = pl.read_parquet(
        Path(cfg.stage_cfg.metadata_input_dir) / "code_metadata.parquet", use_pyarrow=True
    )
    compute_fn = occlude_outliers_fntr(cfg.stage_cfg, code_metadata)

    map_over(cfg, compute_fn=compute_fn)


if __name__ == "__main__":
    main()
