"""Transformations for normalizing MEDS datasets, across both categorical and continuous dimensions."""

import polars as pl

from .. import Stage


@Stage.register
def normalization(
    df: pl.LazyFrame, code_metadata: pl.DataFrame, code_modifiers: list[str] | None = None
) -> pl.LazyFrame:
    """Normalize a MEDS dataset across both categorical and continuous dimensions.

    This function expects a MEDS dataset in flattened form, with columns for:
      - `subject_id`
      - `time`
      - `code`
      - `numeric_value`

    In addition, the `code_metadata` dataset should contain information about the codes in the MEDS dataset,
    including the mandatory columns:
      - `code` (`categorical`)
      - `code/vocab_index` (`int`)
      - Any `code_modifiers` columns, if specified

    Additionally, it must either have:
      - Pre-computed means and standard deviations for the numeric values of the codes in the MEDS dataset,
        via:
        - `values/mean` (`float`)
        - `values/std` (`float`)
      - Or the necessary statistics to compute the per-occurrence mean and standard deviation of the numeric
        values of the codes in the MEDS dataset, via:
        - `values/n_occurrences` (`int`)
        - `values/sum` (`float`)
        - `values/sum_sqd` (`float`)


    The `values/*` functions will be used to normalize the code numeric values to have a mean of 0 and a
    standard deviation of 1. The output dataframe will further be filtered to only contain rows where the
    `code` in the MEDS dataset appears in the `code_metadata` dataset, and the output `code` column will be
    converted to the `code/vocab_index` integral ID from the `code_metadata` dataset.

    This function can further be customized by specifying additional columns to join on, via the
    `code_modifiers` parameter, which must appear in both the MEDS dataset and the code metadata. These
    columns will be discarded from the output dataframe, which will only contain the four expected input
    columns, though normalized.

    Args:
        df: The MEDS dataset to normalize. See above for the expected schema.
        code_metadata: Metadata about the codes in the MEDS dataset. See above for the expected schema.
        code_modifiers: Additional columns to join on, which will be discarded from the output dataframe.

    Returns:
        The normalized MEDS dataset, with the schema described above.

    Examples:
        >>> from datetime import datetime
        >>> MEDS_df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 2, 2, 2, 3],
        ...         "time": [
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...         ],
        ...         "code": ["lab//A", "lab//A", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
        ...         "numeric_value": [1, 3, None, 3, None, None, None],
        ...         "unit": ["mg/dL", "g/dL", None, "mg/dL", None, None, None],
        ...     },
        ...     schema = {
        ...         "subject_id": pl.UInt32,
        ...         "time": pl.Datetime,
        ...         "code": pl.Utf8,
        ...         "numeric_value": pl.Float64,
        ...         "unit": pl.Utf8,
        ...    },
        ... )
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F"],
        ...         "code/vocab_index": [0, 2, 3, 4, 5],
        ...         "values/mean": [2.0, None, None, None, 3],
        ...         "values/std": [0.5, None, None, None, 0.2],
        ...     },
        ...     schema = {
        ...         "code": pl.Utf8,
        ...         "code/vocab_index": pl.UInt32,
        ...         "values/mean": pl.Float64,
        ...         "values/std": pl.Float64,
        ...     },
        ... )
        >>> normalization(MEDS_df.lazy(), code_metadata).collect()
        shape: (6, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ -2.0          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ 2.0           │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 3    ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 0    ┆ 2.0           │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 2    ┆ null          │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 5    ┆ null          │
        └────────────┴─────────────────────┴──────┴───────────────┘
        >>> MEDS_df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 2, 2, 2, 3],
        ...         "time": [
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...         ],
        ...         "code": ["lab//A", "lab//A", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
        ...         "numeric_value": [1, 3, None, 3, None, None, None],
        ...         "unit": ["mg/dL", "g/dL", None, "mg/dL", None, None, None],
        ...     },
        ...     schema = {
        ...         "subject_id": pl.UInt32,
        ...         "time": pl.Datetime,
        ...         "code": pl.Utf8,
        ...         "numeric_value": pl.Float64,
        ...         "unit": pl.Utf8,
        ...    },
        ... )
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["lab//A", "lab//A", "lab//C", "dx//B", "dx//E", "lab//F"],
        ...         "unit": ["mg/dL", "g/dL", None, None, None, None],
        ...         "code/vocab_index": [0, 1, 2, 3, 4, 5],
        ...         "values/mean": [2.0, 3.0, None, None, None, 3],
        ...         "values/std": [0.5, 2.0, None, None, None, 0.2],
        ...     },
        ...     schema = {
        ...         "code": pl.Utf8,
        ...         "unit": pl.Utf8,
        ...         "code/vocab_index": pl.UInt32,
        ...         "values/mean": pl.Float64,
        ...         "values/std": pl.Float64,
        ...     },
        ... )
        >>> normalization(MEDS_df.lazy(), code_metadata, ["unit"]).collect()
        shape: (6, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ -2.0          │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1    ┆ 0.0           │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 3    ┆ null          │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 0    ┆ 2.0           │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 2    ┆ null          │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 5    ┆ null          │
        └────────────┴─────────────────────┴──────┴───────────────┘

        Note that while this function is robust to the inclusion of the default row index column name, it
        doesn't retain any extra columns after the operation. If you want to retain the row index, you should
        file a GitHub issue with this request and we can add it in a future release.
        >>> MEDS_df = MEDS_df.with_columns(pl.lit(1).alias("_row_idx"), pl.lit(2).alias("foobar"))
        >>> normalization(MEDS_df.head(1).lazy(), code_metadata, ["unit"]).collect()
        shape: (1, 4)
        ┌────────────┬─────────────────────┬──────┬───────────────┐
        │ subject_id ┆ time                ┆ code ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---           │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64           │
        ╞════════════╪═════════════════════╪══════╪═══════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ -2.0          │
        └────────────┴─────────────────────┴──────┴───────────────┘
    """

    if code_modifiers is None:
        code_modifiers = []

    cols_to_select = ["code", "code/vocab_index"] + code_modifiers

    mean_col = pl.col("values/sum") / pl.col("values/n_occurrences")
    stddev_col = (pl.col("values/sum_sqd") / pl.col("values/n_occurrences") - mean_col**2) ** 0.5

    code_metadata_columns = set(code_metadata.columns)
    if "values/mean" in code_metadata_columns:
        cols_to_select.append("values/mean")
    else:
        cols_to_select.append(mean_col.alias("values/mean"))

    if "values/std" in code_metadata_columns:
        cols_to_select.append("values/std")
    else:
        cols_to_select.append(stddev_col.alias("values/std"))

    idx_col = "_row_idx"
    df_cols = df.collect_schema().names()
    while idx_col in df_cols:
        idx_col = f"_{idx_col}"

    return (
        df.with_row_index(idx_col)
        .join(
            code_metadata.lazy().select(cols_to_select),
            on=["code"] + code_modifiers,
            how="inner",
            nulls_equal=True,
        )
        .select(
            idx_col,
            "subject_id",
            "time",
            pl.col("code/vocab_index").alias("code"),
            ((pl.col("numeric_value") - pl.col("values/mean")) / pl.col("values/std")).alias("numeric_value"),
        )
        .sort(idx_col)
        .drop(idx_col)
    )
