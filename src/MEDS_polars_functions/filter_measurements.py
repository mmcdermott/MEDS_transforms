"""A polars-to-polars transformation function for filtering patients by sequence length."""

from collections.abc import Callable

import polars as pl

pl.enable_string_cache()
from omegaconf import DictConfig


def filter_codes_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifier_columns: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that filters patient events to only encompass those with a set of permissible codes.

    Args:
        df: The input DataFrame.
        stage_cfg: The configuration for the code filtering stage.

    Returns:
        The processed DataFrame.

    Examples:
        >>> code_metadata_df = pl.DataFrame({
        ...     "code":     pl.Series(["A", "A", "B", "C"], dtype=pl.Categorical),
        ...     "modifier1":          [1,   2,   1,   2],
        ...     "code/n_patients":    [2,   1,   3,   2],
        ...     "code/n_occurrences": [4,   5,   3,   2],
        ... })
        >>> data = pl.DataFrame({
        ...     "patient_id":     [1,   1,   2,   2],
        ...     "code": pl.Series(["A", "B", "A", "C"], dtype=pl.Categorical),
        ...     "modifier1":      [1,   1,   2,   2],
        ... }).lazy()
        >>> stage_cfg = DictConfig({"min_patients_per_code": 2, "min_occurrences_per_code": 3})
        >>> fn = filter_codes_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (2, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ cat  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_patients_per_code": 1, "min_occurrences_per_code": 4})
        >>> fn = filter_codes_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (2, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ cat  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_patients_per_code": 1})
        >>> fn = filter_codes_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (4, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ cat  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        │ 2          ┆ C    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_patients_per_code": None, "min_occurrences_per_code": None})
        >>> fn = filter_codes_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (4, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ cat  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        │ 2          ┆ C    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_occurrences_per_code": 5})
        >>> fn = filter_codes_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (1, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ cat  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 2          ┆ A    ┆ 2         │
        └────────────┴──────┴───────────┘
    """

    min_patients_per_code = stage_cfg.get("min_patients_per_code", None)
    min_occurrences_per_code = stage_cfg.get("min_occurrences_per_code", None)

    filter_exprs = []
    if min_patients_per_code is not None:
        filter_exprs.append(pl.col("code/n_patients") >= min_patients_per_code)
    if min_occurrences_per_code is not None:
        filter_exprs.append(pl.col("code/n_occurrences") >= min_occurrences_per_code)

    if not filter_exprs:
        return lambda df: df

    join_cols = ["code"]
    if code_modifier_columns:
        join_cols.extend(code_modifier_columns)

    allowed_code_metadata = (code_metadata.filter(pl.all_horizontal(filter_exprs)).select(join_cols)).lazy()

    def filter_codes_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        f"""Filters patient events to only encompass those with a set of permissible codes.

        In particular, this function filters the DataFrame to only include (code, modifier) pairs that have
        at least {min_patients_per_code} patients and {min_occurrences_per_code} occurrences.
        """

        idx_col = "_row_idx"
        while idx_col in df.columns:
            idx_col = f"_{idx_col}"

        return (
            df.with_row_count(idx_col)
            .join(allowed_code_metadata, on=join_cols, how="inner")
            .sort(idx_col)
            .drop(idx_col)
        )

    return filter_codes_fn


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
        >>> fn = filter_outliers_fntr(stage_cfg, code_metadata_df, ["modifier1"])
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
    if "values/stddev" not in code_metadata.columns:
        cols_to_select.append(stddev_col.alias("values/stddev"))

    code_metadata = code_metadata.lazy().select(cols_to_select)

    def filter_outliers_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        f"""Filters out outlier numerical values from patient events.

        In particular, this function filters the DataFrame to only include numerical values that are within
        {stddev_cutoff} standard deviations of the mean for the corresponding (code, modifier) pair.
        """

        val = pl.col("numerical_value")
        mean = pl.col("values/mean")
        stddev = pl.col("values/stddev")
        filter_expr = (val - mean).abs() <= stddev_cutoff * stddev

        return (
            df.join(code_metadata, on=join_cols, how="left")
            .with_columns(
                filter_expr.alias("numerical_value/is_inlier"),
                pl.when(filter_expr).then(pl.col("numerical_value")).alias("numerical_value"),
            )
            .drop("values/mean", "values/stddev")
        )

    return filter_outliers_fn
