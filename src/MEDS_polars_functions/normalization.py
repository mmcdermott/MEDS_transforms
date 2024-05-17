"""Transformations for normalizing MEDS datasets, across both categorical and continuous dimensions."""

import polars as pl


def normalize(
    df: pl.LazyFrame, code_metadata: pl.LazyFrame, extra_join_columns: list[str] | None = None
) -> pl.LazyFrame:
    """Normalize a MEDS dataset across both categorical and continuous dimensions.

    This function expects a MEDS dataset in flattened form, with columns for:
      - `patient_id`
      - `timestamp`
      - `code`
      - `numerical_value`

    In addition, the `code_metadata` dataset should contain information about the codes in the MEDS dataset,
    including:
      - `code` (`categorical`)
      - `code/vocab_id` (`int`)
      - `value/mean` (`float`)
      - `value/std` (`float`)

    The `value/*` functions will be used to normalize the code numerical values to have a mean of 0 and a
    standard deviation of 1. The output dataframe will further be filtered to only contain rows where the
    `code` in the MEDS dataset appears in the `code_metadata` dataset, and the output `code` column will be
    converted to the `code/vocab_id` integral ID from the `code_metadata` dataset.

    This function can further be customized by specifying additional columns to join on, via the
    `extra_join_columns` parameter, which must appear in both the MEDS dataset and the code metadata. These
    columns will be discarded from the output dataframe, which will only contain the four expected input
    columns, though normalized.

    Args:
        df: The MEDS dataset to normalize. See above for the expected schema.
        code_metadata: Metadata about the codes in the MEDS dataset. See above for the expected schema.
        extra_join_columns: Additional columns to join on, which will be discarded from the output dataframe.

    Returns:
        The normalized MEDS dataset, with the schema described above.

    Examples:
        >>> from datetime import datetime
        >>> MEDS_df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 2, 2, 2, 3],
        ...         "timestamp": [
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...         ],
        ...         "code": ["lab//A", "lab//A", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
        ...         "numerical_value": [1, 3, None, 3, None, None, None],
        ...         "unit": ["mg/dL", "g/dL", None, "mg/dL", None, None, None],
        ...     },
        ...     schema = {
        ...         "patient_id": pl.UInt32,
        ...         "timestamp": pl.Datetime,
        ...         "code": pl.Categorical(ordering='physical'),
        ...         "numerical_value": pl.Float64,
        ...         "unit": pl.Utf8,
        ...    },
        ... )
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F"],
        ...         "code/vocab_id": [0, 2, 3, 4, 5],
        ...         "value/mean": [2.0, None, None, None, 3],
        ...         "value/std": [0.5, None, None, None, 0.2],
        ...     },
        ...     schema = {
        ...         "code": pl.Categorical(ordering='physical'),
        ...         "code/vocab_id": pl.UInt32,
        ...         "value/mean": pl.Float64,
        ...         "value/std": pl.Float64,
        ...     },
        ... )
        >>> normalize(MEDS_df.lazy(), code_metadata.lazy()).collect()
        shape: (6, 4)
        ┌────────────┬─────────────────────┬──────┬─────────────────┐
        │ patient_id ┆ timestamp           ┆ code ┆ numerical_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---             │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64             │
        ╞════════════╪═════════════════════╪══════╪═════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ -2.0            │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ 2.0             │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 3    ┆ null            │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 0    ┆ 2.0             │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 2    ┆ null            │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 5    ┆ null            │
        └────────────┴─────────────────────┴──────┴─────────────────┘
        >>> MEDS_df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 2, 2, 2, 3],
        ...         "timestamp": [
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...         ],
        ...         "code": ["lab//A", "lab//A", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
        ...         "numerical_value": [1, 3, None, 3, None, None, None],
        ...         "unit": ["mg/dL", "g/dL", None, "mg/dL", None, None, None],
        ...     },
        ...     schema = {
        ...         "patient_id": pl.UInt32,
        ...         "timestamp": pl.Datetime,
        ...         "code": pl.Categorical(ordering='physical'),
        ...         "numerical_value": pl.Float64,
        ...         "unit": pl.Utf8,
        ...    },
        ... )
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["lab//A", "lab//A", "lab//C", "dx//B", "dx//E", "lab//F"],
        ...         "unit": ["mg/dL", "g/dL", None, None, None, None],
        ...         "code/vocab_id": [0, 1, 2, 3, 4, 5],
        ...         "value/mean": [2.0, 3.0, None, None, None, 3],
        ...         "value/std": [0.5, 2.0, None, None, None, 0.2],
        ...     },
        ...     schema = {
        ...         "code": pl.Categorical(ordering='physical'),
        ...         "unit": pl.Utf8,
        ...         "code/vocab_id": pl.UInt32,
        ...         "value/mean": pl.Float64,
        ...         "value/std": pl.Float64,
        ...     },
        ... )
        >>> normalize(MEDS_df.lazy(), code_metadata.lazy(), ["unit"]).collect()
        shape: (6, 4)
        ┌────────────┬─────────────────────┬──────┬─────────────────┐
        │ patient_id ┆ timestamp           ┆ code ┆ numerical_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---             │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64             │
        ╞════════════╪═════════════════════╪══════╪═════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ -2.0            │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1    ┆ 0.0             │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 3    ┆ null            │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 0    ┆ 2.0             │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 2    ┆ null            │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 5    ┆ null            │
        └────────────┴─────────────────────┴──────┴─────────────────┘
    """

    if extra_join_columns is None:
        extra_join_columns = []

    return df.join(
        code_metadata,
        on=["code"] + extra_join_columns,
        how="inner",
        join_nulls=True,
    ).select(
        "patient_id",
        "timestamp",
        pl.col("code/vocab_id").alias("code"),
        ((pl.col("numerical_value") - pl.col("value/mean")) / pl.col("value/std")).alias("numerical_value"),
    )
