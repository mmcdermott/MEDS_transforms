"""Convert numeric values into categorical code modifiers through pre-computed or specified bins."""

import logging

import polars as pl

from .. import Stage

logger = logging.getLogger(__name__)


def process_quantiles(df: pl.DataFrame) -> pl.DataFrame:
    """Process quantiles in a DataFrame, using custom quantiles if available.

    Args:
        df: A Polars DataFrame containing columns for values/quantiles and optionally custom_quantiles

    Returns:
        A DataFrame with processed quantiles and updated code column

    Examples:
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "code": ["lab//A", "lab//B", "lab//C", "lab//D"],
    ...     "numeric_value": [-1.0, 2.0, None, 0.0],
    ...     "values/quantiles": [
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3}
    ...     ],
    ...     "custom_quantiles": [None,
    ...         {"values/quantile/0.5": 1.5},
    ...         None,
    ...         None
    ...     ],
    ...     "code/vocab_index": [0, 1, 2, 3],
    ... })
    >>> result = process_quantiles(df)
    >>> result.select(["code", "numeric_value"])
    shape: (4, 2)
    ┌──────────────┬───────────────┐
    │ code         ┆ numeric_value │
    │ ---          ┆ ---           │
    │ str          ┆ f64           │
    ╞══════════════╪═══════════════╡
    │ lab//A//_Q_1 ┆ -1.0          │
    │ lab//B//_Q_2 ┆ 2.0           │
    │ lab//C       ┆ null          │
    │ lab//D//_Q_1 ┆ 0.0           │
    └──────────────┴───────────────┘
    >>> # Test with only custom quantiles
    >>> df_custom = pl.DataFrame({
    ...     "code": ["lab//A", "lab//A", "lab//A"],
    ...     "numeric_value": [-0.5, 1.0, 4.0],
    ...     "values/quantiles": [
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...         {"values/quantile/0.2": 0, "values/quantile/0.4": 1,
    ...          "values/quantile/0.6": 2, "values/quantile/0.8": 3},
    ...     ],
    ...     "custom_quantiles": [
    ...         None,
    ...         None,
    ...         None,
    ...     ],
    ...     "code/vocab_index": [0, 0, 0],
    ... })
    >>> result_custom = process_quantiles(df_custom)
    >>> result_custom.select(["code", "numeric_value"])
    shape: (3, 2)
    ┌──────────────┬───────────────┐
    │ code         ┆ numeric_value │
    │ ---          ┆ ---           │
    │ str          ┆ f64           │
    ╞══════════════╪═══════════════╡
    │ lab//A//_Q_1 ┆ -0.5          │
    │ lab//A//_Q_2 ┆ 1.0           │
    │ lab//A//_Q_5 ┆ 4.0           │
    └──────────────┴───────────────┘
    """
    # Use custom_quantiles if available, otherwise use values/quantiles
    df = df.with_columns(
        effective_quantiles=pl.when(pl.col("custom_quantiles").is_not_null())
        .then(pl.col("custom_quantiles"))
        .otherwise(pl.col("values/quantiles"))
    )

    # Unnest the effective_quantiles and calculate the quantile
    quantile_columns = pl.selectors.starts_with("values/quantile/")
    df = df.unnest("effective_quantiles").with_columns(
        quantile=pl.when(pl.col("numeric_value").is_not_null()).then(
            pl.sum_horizontal(quantile_columns.lt(pl.col("numeric_value"))).add(1)
        )
    )

    # Create the new code with quantile information
    code_quantile_concat = pl.concat_str(pl.col("code"), pl.lit("//_Q_"), pl.col("quantile"))
    df = df.with_columns(
        code=pl.when(pl.col("quantile").is_not_null()).then(code_quantile_concat).otherwise(pl.col("code"))
    )

    # Clean up intermediate columns
    df = df.drop("quantile", "values/quantiles", "custom_quantiles", quantile_columns)

    return df


@Stage.register
def bin_numeric_values_fntr(
    stage_cfg,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
) -> pl.LazyFrame:
    """Converts the numeric values in a MEDS dataset to discrete quantiles that are added to the code name.

    Returns:
        - A new DataFrame with the quantile values added to the code name.
        - A new DataFrame with the metadata for the quantile codes.

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
    ...         "code": ["lab//A", "lab//C", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
    ...         "numeric_value": [1, 3, None, 3, None, None, None],
    ...     },
    ...     schema = {
    ...         "subject_id": pl.UInt32,
    ...         "time": pl.Datetime,
    ...         "code": pl.Utf8,
    ...         "numeric_value": pl.Float64,
    ...    },
    ... )
    >>> code_metadata = pl.DataFrame(
    ...     {
    ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F", "dx//D"],
    ...         "values/quantiles": [ # [[-3,-1,1,3], [-3,-1,1,3], [], [-3,-1,1,3], [-3,-1,1,3], [-3,-1,1,3]],
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": None, "values/quantile/0.4": None,
    ...                 "values/quantile/0.6": None, "values/quantile/0.8": None},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...             {"values/quantile/0.2": -3, "values/quantile/0.4": -1,
    ...                 "values/quantile/0.6": 1, "values/quantile/0.8": 3},
    ...         ],
    ...     },
    ...     schema = {
    ...         "code": pl.Utf8,
    ...         "values/quantiles": pl.Struct([
    ...             pl.Field("values/quantile/0.2", pl.Float64),
    ...             pl.Field("values/quantile/0.4", pl.Float64),
    ...             pl.Field("values/quantile/0.6", pl.Float64),
    ...             pl.Field("values/quantile/0.8", pl.Float64),
    ...         ]), # pl.List(pl.Float64),
    ...     },
    ... )
    >>> custom_quantiles = {"lab//C": {"values/quantile/0.5": 0}}
    >>> fn = bin_numeric_values_fntr({"custom_quantiles": custom_quantiles}, code_metadata)
    >>> fn(MEDS_df).sort("subject_id", "time", "code")
    shape: (7, 4)
    ┌────────────┬─────────────────────┬──────────────┬───────────────┐
    │ subject_id ┆ time                ┆ code         ┆ numeric_value │
    │ ---        ┆ ---                 ┆ ---          ┆ ---           │
    │ u32        ┆ datetime[μs]        ┆ str          ┆ f64           │
    ╞════════════╪═════════════════════╪══════════════╪═══════════════╡
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_3 ┆ 1.0           │
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//C//_Q_2 ┆ 3.0           │
    │ 1          ┆ 2021-01-02 00:00:00 ┆ dx//B        ┆ null          │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ dx//D        ┆ null          │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//A//_Q_4 ┆ 3.0           │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//C       ┆ null          │
    │ 3          ┆ 2022-10-02 00:00:00 ┆ lab//F       ┆ null          │
    └────────────┴─────────────────────┴──────────────┴───────────────┘
    >>> custom_quantiles = {"lab//A": {"values/quantile/0.5": 3}}
    >>> fn = bin_numeric_values_fntr({"custom_quantiles": custom_quantiles}, code_metadata)
    >>> fn(MEDS_df).sort("subject_id", "time", "code")
    shape: (7, 4)
    ┌────────────┬─────────────────────┬──────────────┬───────────────┐
    │ subject_id ┆ time                ┆ code         ┆ numeric_value │
    │ ---        ┆ ---                 ┆ ---          ┆ ---           │
    │ u32        ┆ datetime[μs]        ┆ str          ┆ f64           │
    ╞════════════╪═════════════════════╪══════════════╪═══════════════╡
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//A//_Q_1 ┆ 1.0           │
    │ 1          ┆ 2021-01-01 00:00:00 ┆ lab//C//_Q_4 ┆ 3.0           │
    │ 1          ┆ 2021-01-02 00:00:00 ┆ dx//B        ┆ null          │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ dx//D        ┆ null          │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//A//_Q_1 ┆ 3.0           │
    │ 2          ┆ 2022-10-02 00:00:00 ┆ lab//C       ┆ null          │
    │ 3          ┆ 2022-10-02 00:00:00 ┆ lab//F       ┆ null          │
    └────────────┴─────────────────────┴──────────────┴───────────────┘
    """
    if code_modifiers is None:
        code_modifiers = []

    # Step 1: Add custom_quantiles column to code_metadata
    custom_quantiles = stage_cfg.get("custom_quantiles", {})

    if custom_quantiles:
        custom_quantiles_series = pl.Series(
            name="custom_quantiles",
            values=[
                custom_quantiles.get(code) if code is not None else None for code in code_metadata["code"]
            ],
        )
        code_metadata = code_metadata.with_columns(custom_quantiles_series)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return process_quantiles(
            df.join(code_metadata, on=["code", *code_modifiers], how="left", maintain_order="left")
        )

    return fn
