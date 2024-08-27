#!/usr/bin/env python
"""Functions for tensorizing MEDS datasets."""

from functools import partial

import hydra
import polars as pl
from loguru import logger
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.mapreduce.utils import shard_iterator


def convert_to_NRT(df: pl.LazyFrame) -> JointNestedRaggedTensorDict:
    """This converts a tokenized dataframe into a nested ragged tensor.

    Most of the work for this function is actually done in `tokenize` -- this function is just a wrapper
    to convert the output into a nested ragged tensor using polars' built-in `to_dict` method.

    Args:
        df: The tokenized dataframe.

    Returns:
        A `JointNestedRaggedTensorDict` object representing the tokenized dataframe, accounting for however
        many levels of ragged nesting are present among the codes and numeric values.

    Raises:
        ValueError: If there are no time delta columns or if there are multiple time delta columns.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 2],
        ...     "time_delta_days": [[float("nan"), 12.0], [float("nan")]],
        ...     "code": [[[101.0, 102.0], [103.0]], [[201.0, 202.0]]],
        ...     "numeric_value": [[[2.0, 3.0], [4.0]], [[6.0, 7.0]]]
        ... })
        >>> df
        shape: (2, 4)
        ┌────────────┬─────────────────┬───────────────────────────┬─────────────────────┐
        │ patient_id ┆ time_delta_days ┆ code                      ┆ numeric_value       │
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
        numeric_value
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
    time_delta_cols = [c for c in df.collect_schema().names() if c.startswith("time_delta_")]

    if len(time_delta_cols) == 0:
        raise ValueError("Expected at least one time delta column, found none")
    elif len(time_delta_cols) > 1:
        raise ValueError(f"Expected exactly one time delta column, found columns: {time_delta_cols}")

    time_delta_col = time_delta_cols[0]

    tensors_dict = df.select(time_delta_col, "code", "numeric_value").collect().to_dict(as_series=False)

    if all((not v) for v in tensors_dict.values()):
        logger.warning("All columns are empty. Returning an empty tensor dict.")
        return JointNestedRaggedTensorDict({})

    for k, v in tensors_dict.items():
        if not v:
            raise ValueError(f"Column {k} is empty")

    return JointNestedRaggedTensorDict(tensors_dict)


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    map_over(
        cfg,
        compute_fn=convert_to_NRT,
        write_fn=JointNestedRaggedTensorDict.save,
        shard_iterator_fntr=partial(shard_iterator, in_prefix="event_seqs/", out_suffix=".nrt"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
