"""Functions for tokenizing MEDS datasets.

Here, _tokenization_ refers specifically to the process of converting a longitudinal, irregularly sampled,
continuous time sequence into a temporal sequence at the level that will be consumed by deep-learning models.

All these functions take in _normalized_ data -- meaning data where there are _no longer_ any code modifiers,
as those have been normalized alongside codes into integer indices (in the output code column). The only
columns of concern here thus are `patient_id`, `timestamp`, `code`, `numerical_value`.
"""

import polars as pl

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * 60.0
SECONDS_PER_DAY = SECONDS_PER_HOUR * 24.0

def fill_to_nans(col: str | pl.Expr) -> pl.Expr:
    """This function fills infinite and null values with NaN.

    This enables the downstream functions to naturally tensorize data into numpy or Torch tensors.

    Args:
        col: The input column.

    Returns:
        A `pl.Expr` object that fills infinite and null values with NaN.

    Examples:
        >>> raise NotImplementedError
    """

    if isinstance(col, str):
        col = pl.col(col)

    return (
        pl.when(col.is_infinite() | col.is_null())
        .then(float('nan'))
        .otherwise(col)
        .keep_name()
    )

def split_static_and_dynamic(df: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """This function splits the input data into static and dynamic data.

    Static data is data that has a null timestamp, and dynamic data is everything else.

    Args:
        df: The input data.

    Returns:
        A tuple of two `pl.LazyFrame` objects, the first being the static data and the second being the
        dynamic data.

    Examples:
        >>> raise NotImplementedError
    """

    static = df.filter(pl.col("timestamp").is_null())
    dynamic = df.filter(pl.col("timestamp").is_not_null())
    return static, dynamic

def extract_statics_and_schema(df: pl.LazyFrame) -> pl.LazyFrame:
    """This function extracts static data and schema information (sequence of patient unique timestamps).

    Args:
        df: The input data.

    Returns:
        A `pl.LazyFrame` object containing the static data and the unique timestamps of the patient, grouped
        by patient as lists, in the same order as the patient IDs occurred in the original file.
    """

    static, dynamic = split_static_and_dynamic(df)

    # This collects static data by patient ID and stores only (as a list) the codes and numerical values.
    static_by_patient = static.group_by("patient_id", maintain_order=True).agg("code", "numerical_value")

    # This collects the unique timestamps for each patient.
    schema_by_patient = (
        dynamic
        .group_by("patient_id", maintain_order=True)
        .agg(
            pl.col("timestamp").min().alias("start_time"),
            pl.col("timestamp").unique(maintain_order=True)
        )
    )

    return (
        static_by_patient
        .join(schema_by_patient, on="patient_id", how="inner")
        .with_row_index("patient_offset")
    )


def extract_seq_of_patient_events(df: pl.LazyFrame) -> pl.LazyFrame:
    """This function extracts sequences of patient events, which are sequences of measurements.

    The result of this can be naturally tensorized into a `JointNestedRaggedTensorDict` object.

    Args:
        df: The input data.

    Returns:
        A `pl.LazyFrame` object containing the sequences of patient events, with the following columns:
            - `patient_id`: The patient ID.
            - `time_delta/days`: The time delta in days, as a list of floats (ragged).
            - `code`: The code, as a list of lists of ints (ragged in both levels).
            - `numerical_value`: The numerical value as a list of lists of floats (ragged in both levels).

    Examples:
        >>> raise NotImplementedError
    """

    _, dynamic = split_static_and_dynamic(df)

    time_delta_days_expr = (pl.col("timestamp").diff().dt.total_seconds() / SECONDS_PER_DAY).cast(pl.Float64)

    return (
        dynamic
        .group_by("patient_id", "timestamp", maintain_order=True)
        .agg(fill_to_nans("code"), fill_to_nans("numerical_value"))
        .group_by("patient_id", maintain_order=True)
        .agg(
            fill_to_nans(time_delta_days_expr).alias("time_delta/days"),
            "code",
            "numerical_value",
        )
    )
