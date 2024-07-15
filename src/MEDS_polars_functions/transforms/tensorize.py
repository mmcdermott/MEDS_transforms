#!/usr/bin/env python
"""Functions for tensorizing MEDS datasets.

TODO
"""

from functools import partial
from importlib.resources import files

import hydra
import polars as pl
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig

from MEDS_polars_functions.mapreduce.mapper import map_over
from MEDS_polars_functions.mapreduce.utils import shard_iterator


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
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 2],
        ...     "time_delta_days": [[float("nan"), 12.0], [float("nan")]],
        ...     "code": [[[101.0, 102.0], [103.0]], [[201.0, 202.0]]],
        ...     "numerical_value": [[[2.0, 3.0], [4.0]], [[6.0, 7.0]]]
        ... })
        >>> df
        shape: (2, 4)
        ┌────────────┬─────────────────┬───────────────────────────┬─────────────────────┐
        │ patient_id ┆ time_delta_days ┆ code                      ┆ numerical_value     │
        │ ---        ┆ ---             ┆ ---                       ┆ ---                 │
        │ i64        ┆ list[f64]       ┆ list[list[f64]]           ┆ list[list[f64]]     │
        ╞════════════╪═════════════════╪═══════════════════════════╪═════════════════════╡
        │ 1          ┆ [NaN, 12.0]     ┆ [[101.0, 102.0], [103.0]] ┆ [[2.0, 3.0], [4.0]] │
        │ 2          ┆ [NaN]           ┆ [[201.0, 202.0]]          ┆ [[6.0, 7.0]]        │
        └────────────┴─────────────────┴───────────────────────────┴─────────────────────┘
        >>> nrt = convert_to_NRT(df.lazy())
        >>> for k, v in sorted(list(nrt.to_dense().items())):
        ...     print(k)
        ...     print(v)
        code
        [[[101. 102.]
          [103.   0.]]
        <BLANKLINE>
         [[201. 202.]
          [  0.   0.]]]
        dim1/mask
        [[ True  True]
         [ True False]]
        dim2/mask
        [[[ True  True]
          [ True False]]
        <BLANKLINE>
         [[ True  True]
          [False False]]]
        numerical_value
        [[[2. 3.]
          [4. 0.]]
        <BLANKLINE>
         [[6. 7.]
          [0. 0.]]]
        time_delta_days
        [[nan 12.]
         [nan  0.]]
    """

    # There should only be one time delta column, but this ensures we catch it regardless of the unit of time
    # used to convert the time deltas, and that we verify there is only one such column.
    time_delta_cols = [c for c in tokenized_df.collect_schema().names() if c.startswith("time_delta_")]

    if len(time_delta_cols) == 0:
        raise ValueError("Expected at least one time delta column, found none")
    elif len(time_delta_cols) > 1:
        raise ValueError(f"Expected exactly one time delta column, found columns: {time_delta_cols}")

    time_delta_col = time_delta_cols[0]

    return JointNestedRaggedTensorDict(
        tokenized_df.select(time_delta_col, "code", "numerical_value").collect().to_dict(as_series=False)
    )


config_yaml = files("MEDS_polars_functions").joinpath("configs/preprocess.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """TODO."""

    map_over(
        cfg,
        compute_fn=convert_to_NRT,
        output_fn=JointNestedRaggedTensorDict.save,
        shard_iterator_fntr=partial(shard_iterator, in_prefix="event_seqs/", out_suffix=".nrt"),
    )


if __name__ == "__main__":
    main()
