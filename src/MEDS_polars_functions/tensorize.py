"""Functions for tensorizing MEDS datasets.

TODO
"""

from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
import polars as pl

def convert_to_NRT(tokenized_df: pl.LazyFrame) -> JointNestedRaggedTensorDict:
    """This converts a tokenized dataframe into a nested ragged tensor.

    Most of the work for this function is actually done in `tokenize` -- this function is just a wrapper
    to convert the output into a nested ragged tensor using polars' built-in `to_dict` method.

    Args:
        tokenized_df: The tokenized dataframe.

    Returns:
        A `JointNestedRaggedTensorDict` object representing the tokenized dataframe, accounting for however
        many levels of ragged nesting are present among the codes and numerical values.

    Raises:
        ValueError: If there are no time delta columns or if there are multiple time delta columns.

    Examples:
        >>> raise NotImplementedError
    """

    # There should only be one time delta column, but this ensures we catch it regardless of the unit of time
    # used to convert the time deltas, and that we verify there is only one such column.
    time_delta_cols = [c for c in tokenized_df.columns if c.startswith("time_delta/")]

    if len(time_delta_cols) == 0:
        raise ValueError("Expected at least one time delta column, found none")
    elif len(time_delta_cols) > 1:
        raise ValueError(f"Expected exactly one time delta column, found columns: {time_delta_cols}")

    time_delta_col = time_delta_cols[0]

    return JointNestedRaggedTensorDict(
        tokenized_df.select(
            time_delta_col,
            "code",
            "numerical_value").collect().to_dict(as_series=False)
    )
