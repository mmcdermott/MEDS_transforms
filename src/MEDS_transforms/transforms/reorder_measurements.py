#!/usr/bin/env python
"""A polars-to-polars transformation function for filtering patients by sequence length."""
from collections.abc import Callable

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, ListConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.utils import get_smallest_valid_uint_type


def reorder_by_code_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifiers: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Re-orders a dataframe within the temporal and subject ID ordering via a specified code order.

    Args:
        stage_cfg: TODO
        code_metadata: TODO
        code_modifiers: TODO

    Returns:
        A function with signature `Callable[[pl.LazyFrame], pl.LazyFrame]` that re-orders the input DataFrame.

    Examples:
        >>> code_metadata_df = pl.DataFrame({"code": ["A", "A", "B", "C"], "modifier1": [1, 2, 1, 2]})
        >>> data = pl.DataFrame({
        ...     "patient_id":[1, 1, 2, 2], "time": [1, 1, 1, 1],
        ...     "code": ["A", "B", "A", "C"], "modifier1": [1, 2, 1, 2]
        ... })
        >>> stage_cfg = DictConfig({"ordered_code_patterns": ["B", "A"]})
        >>> fn = reorder_by_code_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data.lazy()).collect()
        shape: (4, 4)
        ┌────────────┬──────┬──────┬───────────┐
        │ patient_id ┆ time ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---  ┆ ---       │
        │ i64        ┆ i64  ┆ str  ┆ i64       │
        ╞════════════╪══════╪══════╪═══════════╡
        │ 1          ┆ 1    ┆ B    ┆ 2         │
        │ 1          ┆ 1    ┆ A    ┆ 1         │
        │ 2          ┆ 1    ┆ A    ┆ 1         │
        │ 2          ┆ 1    ┆ C    ┆ 2         │
        └────────────┴──────┴──────┴───────────┘
        >>> code_metadata_df = pl.DataFrame({
        ...     "code": ["LAB//foo", "ADMISSION//bar", "LAB//baz", "ADMISSION//qux", "DISCHARGE"],
        ... })
        >>> data = pl.DataFrame({
        ...     "patient_id":[1, 1, 1, 2, 2, 2],
        ...     "time": [1, 1, 1, 1, 2, 3],
        ...     "code": ["LAB//foo", "ADMISSION//bar", "LAB//baz", "ADMISSION//qux", "DISCHARGE", "LAB//baz"],
        ... })
        >>> stage_cfg = DictConfig({
        ...     "ordered_code_patterns": ["ADMISSION.*", "LAB//baza", "LAB//f$", "LAB//b.*", "DISCHARGE"]
        ... })
        >>> fn = reorder_by_code_fntr(stage_cfg, code_metadata_df)
        >>> fn(data.lazy()).collect()
        shape: (6, 3)
        ┌────────────┬──────┬────────────────┐
        │ patient_id ┆ time ┆ code           │
        │ ---        ┆ ---  ┆ ---            │
        │ i64        ┆ i64  ┆ str            │
        ╞════════════╪══════╪════════════════╡
        │ 1          ┆ 1    ┆ ADMISSION//bar │
        │ 1          ┆ 1    ┆ LAB//baz       │
        │ 1          ┆ 1    ┆ LAB//foo       │
        │ 2          ┆ 1    ┆ ADMISSION//qux │
        │ 2          ┆ 2    ┆ DISCHARGE      │
        │ 2          ┆ 3    ┆ LAB//baz       │
        └────────────┴──────┴────────────────┘
        >>> fn = reorder_by_code_fntr({}, code_metadata_df)
        >>> fn(data.lazy()).collect()
        shape: (6, 3)
        ┌────────────┬──────┬────────────────┐
        │ patient_id ┆ time ┆ code           │
        │ ---        ┆ ---  ┆ ---            │
        │ i64        ┆ i64  ┆ str            │
        ╞════════════╪══════╪════════════════╡
        │ 1          ┆ 1    ┆ LAB//foo       │
        │ 1          ┆ 1    ┆ ADMISSION//bar │
        │ 1          ┆ 1    ┆ LAB//baz       │
        │ 2          ┆ 1    ┆ ADMISSION//qux │
        │ 2          ┆ 2    ┆ DISCHARGE      │
        │ 2          ┆ 3    ┆ LAB//baz       │
        └────────────┴──────┴────────────────┘
        >>> fn = reorder_by_code_fntr({"ordered_code_patterns": "foo"}, code_metadata_df)
        Traceback (most recent call last):
            ...
        ValueError: The 'ordered_code_patterns' field must be a list of strings. Got foo.
        >>> fn = reorder_by_code_fntr({"ordered_code_patterns": [32]}, code_metadata_df)
        Traceback (most recent call last):
            ...
        ValueError: Each element of 'ordered_code_patterns' must be a string. Got 32.
    """

    ordered_code_patterns = stage_cfg.get("ordered_code_patterns", None)
    if not ordered_code_patterns:
        return lambda df: df

    if not isinstance(ordered_code_patterns, (list, ListConfig)):
        raise ValueError(
            f"The 'ordered_code_patterns' field must be a list of strings. Got {ordered_code_patterns}."
        )

    num_code_patterns = len(ordered_code_patterns)
    code_pattern_idx_dtype = get_smallest_valid_uint_type(num_code_patterns + 1)  # TODO: make function

    join_cols = ["code"]
    if code_modifiers:
        logger.warning("Code reordering currently only matches against the 'code' column, not code modifiers")
        join_cols.extend(code_modifiers)

    cols_to_select = ["code"]
    if code_modifiers:
        cols_to_select.extend(code_modifiers)

    code_order_idx_exprs = pl  # this let's us chain the when/then/otherwise calls equivalently over the loop.
    for i, code_matcher in enumerate(ordered_code_patterns):
        if not isinstance(code_matcher, str):
            raise ValueError(f"Each element of 'ordered_code_patterns' must be a string. Got {code_matcher}.")
        logger.debug(f"Creating code order index expression @ {i} for code pattern {code_matcher}")
        matcher_expr = pl.col("code").str.contains(code_matcher)
        idx_expr = pl.lit(i, dtype=code_pattern_idx_dtype)
        code_order_idx_exprs = code_order_idx_exprs.when(matcher_expr).then(idx_expr)

    code_order_idx_exprs = code_order_idx_exprs.otherwise(num_code_patterns).alias("code_order_idx")

    code_indices = code_metadata.lazy().select(*join_cols, code_order_idx_exprs)

    def reorder_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Reorders the measurements within each event (unique timepoint) by the specified code order."""

        return (
            df.join(code_indices, on=join_cols, how="left", coalesce=True)
            .sort("patient_id", "time", "code_order_idx", maintain_order=True)
            .drop("code_order_idx")
        )

    return reorder_fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Reorders measurements within each patient event (unique timepoint) by the specified code order.

    In particular, given a set of [regex crate compatible](https://docs.rs/regex/latest/regex/) regexes in the
    `stage_cfg.ordered_code_patterns` list, this script will re-order the measurements within each event
    (unique timepoint) such that the measurements are sorted by the index of the first regex that matches
    their code in the `ordered_code_patterns` list. So, if the `ordered_code_patterns` list is
    `["foo$", "bar", "foo.*"]`, and a single patient event has measurements with codes
    `["foobar", "barbaz", "foo", "quat"]`, the measurements will be re-ordered to the order:
    `["foo", "foobar", "barbaz", "quat"]`, because:
      - "foo" matches the first regex in the list (the `foo$` matches any string with "foo" at the end).
      - "foobar" matches the second regex in the list (the `bar` matches any string with "bar" in it), and
        "foobar" is the first measurement in the original order that matches this regex.
      - "barbaz" matches the second regex in the list.
      - "quat" matches no regex in the list, so it occurs after any measurement that matches a regex.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file. It cannot be overwritten
    directly on the command line, but can be overwritten implicitly by overwriting components of the
    `stage_configs.extract_code_metadata` key.

    Args:
        stage_configs.reorder_measurements.ordered_code_patterns: A list of regex patterns that specify the
            order of the codes within each patient event (unique timepoint). To specify this on the command
            line, use the hydra list syntax by enclosing the entire key-value string argument in single
            quotes: ``'stage_configs.reorder_measurements.ordered_code_patterns=["foo$", "bar", "foo.*"]'``.
    """

    map_over(cfg, reorder_by_code_fntr)


if __name__ == "__main__":
    main()
