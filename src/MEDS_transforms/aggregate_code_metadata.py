"""Utilities for grouping and/or reducing MEDS cohort files by code to collect metadata properties."""

import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import hydra
import polars as pl
import polars.selectors as cs
from meds import subject_id_field
from omegaconf import DictConfig, ListConfig, OmegaConf

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce import is_complete_parquet_file, map_over
from MEDS_transforms.utils import write_lazyframe

logger = logging.getLogger(__name__)


class METADATA_FN(StrEnum):
    """Enumeration of metadata functions that can be applied to a group of codes.

    This enumeration contains the supported code-metadata collection and aggregation function names that can
    be applied to codes (or, rather, unique code & modifier units) in a MEDS cohort. Each function name is
    mapped, in the below `CODE_METADATA_AGGREGATIONS` dictionary, to mapper and reducer functions that (a)
    collect the raw data at a per code-modifier level from MEDS subject-level shards and (b) aggregates two or
    more per-shard metadata files into a single metadata file, which can be used to merge metadata across all
    shards into a single file.

    Note that, by design, these aggregations are all those that permit simple, single-variable reductions.
    E.g., rather than tracking the mean and standard deviation of a numeric value, we track the sum of the
    values and the sum of the squares of the values. This is because the mean and standard deviation can be
    trivially calculated from these two values, but ``sum`` and ``sum_sqd`` both can be reduced with a simple
    summation operation across shards, whereas the mean and standard deviation would require a more complex
    reduction function. This convention is _not_ motivated by any performance concerns -- such implications
    would be negligible -- but rather by a desire to keep the aggregation functions simple and easy to
    understand and verify, and by the reality that it is highly unlikely that one would ever care about the
    mean and standard deviation of a value on a per-shard, rather than per-dataset basis, for example, so the
    extra calculation being saved for after the full reduction process does not impact usability.

    These are stored as a `StrEnum` so that they can be easily specified by the user in a configuration file
    or on the command line.

    Args:
        "code/n_subjects": Collects the number of unique subjects who have (anywhere in their record) the code
            & modifiers group.
        "code/n_occurrences": Collects the total number of occurrences of the code & modifiers group across
            all observations for all subjects.
        "values/n_subjects": Collects the number of unique subjects who have a non-null, non-nan
            numeric_value field for the code & modifiers group.
        "values/n_occurrences": Collects the total number of non-null, non-nan numeric_value occurrences for
            the code & modifiers group across all observations for all subjects.
        "values/n_ints": Collects the number of times the observed, non-null numeric_value for the code &
            modifiers group is an integral value (i.e., a whole number, not an integral type).
        "values/sum": Collects the sum of the non-null, non-nan numeric_value values for the code &
            modifiers group.
        "values/sum_sqd": Collects the sum of the squares of the non-null, non-nan numeric_value values for
            the code
        "values/min": Collects the minimum non-null, non-nan numeric_value value for the code & modifiers
        "values/max": Collects the maximum non-null, non-nan numeric_value value for the code & modifiers
        "values/quantiles": Collects the specified quantiles over all observed numeric values for the code &
            modifiers group. The quantiles are specified in the output as a polars struct field, with the
            quantile as the key and the value as the quantile value. The desired quantiles are specified in
            the configuration file using the dictionary syntax for the aggregation.
    """

    CODE_N_PATIENTS = "code/n_subjects"
    CODE_N_OCCURRENCES = "code/n_occurrences"
    VALUES_N_PATIENTS = "values/n_subjects"
    VALUES_N_OCCURRENCES = "values/n_occurrences"
    VALUES_N_INTS = "values/n_ints"
    VALUES_SUM = "values/sum"
    VALUES_SUM_SQD = "values/sum_sqd"
    VALUES_MIN = "values/min"
    VALUES_MAX = "values/max"
    VALUES_QUANTILES = "values/quantiles"


class MapReducePair(NamedTuple):
    """A named tuple pair of a mapper and reducer function for type safety and clarity.

    The mapper (element 0) should be a polars expression that aggregates the data for a single metadata
    aggregation function. It must be suitable for use in a group_by operation on a polars LazyFrame. The
    reducer (element 1) should be a function that takes either a single polars expression, a sequence of
    polars expressions, or a polars Selector and reduces them into a single polars expression, which is
    returned. Built in functions such as `pl.sum_horizontal` and `pl.max_horizontal` are examples of functions
    that can be used as reducers. Custom defined functions can also be used.

    There is no validation of these criteria in the constructor, so it is up to the user to ensure that the
    functions provided are suitable for their intended use.

    Args:
        mapper: A polars expression that aggregates the data for a single metadata aggregation function. It
            must be suitable for use in a group_by operation on a polars LazyFrame.
        reducer: A function that takes either a single polars expression, a sequence of polars expressions, or
            a polars Selector and reduces them into a single polars expression, which is returned.
    """

    mapper: pl.Expr
    reducer: Callable[[pl.Expr | Sequence[pl.Expr] | cs._selector_proxy_], pl.Expr]


def quantile_reducer(cols: cs._selector_proxy_, quantiles: list[float]) -> pl.Expr:
    """Calculates the specified quantiles for the combined set of all numerical values in `cols`.

    Args:
        cols: A polars selector that selects the column(s) containing the numerical values for which the
            quantiles should be calculated.
        quantiles: A list of floats specifying the quantiles that should be calculated.

    Returns:
        A polars expression that calculates the specified quantiles for the combined set of all numerical
        values in `cols`.

    Examples:
        >>> df = pl.DataFrame({
        ...     "key": [1, 2],
        ...     "vals/shard1": [[1, 2, float('nan')], [None, 3]],
        ...     "vals/shard2": [[3.0, 4], [30]],
        ... }, strict=False)
        >>> expr = quantile_reducer(cs.starts_with("vals/"), [0.01, 0.5, 0.75])
        >>> df.select(expr)
        shape: (1, 1)
        ┌──────────────────┐
        │ values/quantiles │
        │ ---              │
        │ struct[3]        │
        ╞══════════════════╡
        │ {1.0,3.0,30.0}   │
        └──────────────────┘
        >>> df.select("key", expr.over("key"))
        shape: (2, 2)
        ┌─────┬──────────────────┐
        │ key ┆ values/quantiles │
        │ --- ┆ ---              │
        │ i64 ┆ struct[3]        │
        ╞═════╪══════════════════╡
        │ 1   ┆ {1.0,3.0,4.0}    │
        │ 2   ┆ {3.0,30.0,30.0}  │
        └─────┴──────────────────┘
    """

    vals = pl.concat_list(cols.fill_null([])).explode()

    quantile_cols = [f"values/quantile/{q}" for q in quantiles]
    quantiles_struct = {col: vals.quantile(q).alias(col) for col, q in zip(quantile_cols, quantiles)}

    return pl.struct(**quantiles_struct).alias(METADATA_FN.VALUES_QUANTILES)


VAL = pl.col("numeric_value")
VAL_PRESENT: pl.Expr = VAL.is_not_null() & VAL.is_not_nan()
IS_INT: pl.Expr = VAL.round() == VAL
PRESENT_VALS = VAL.filter(VAL_PRESENT)

CODE_METADATA_AGGREGATIONS: dict[METADATA_FN, MapReducePair] = {
    METADATA_FN.CODE_N_PATIENTS: MapReducePair(pl.col(subject_id_field).n_unique(), pl.sum_horizontal),
    METADATA_FN.CODE_N_OCCURRENCES: MapReducePair(pl.len(), pl.sum_horizontal),
    METADATA_FN.VALUES_N_PATIENTS: MapReducePair(
        pl.col(subject_id_field).filter(VAL_PRESENT).n_unique(), pl.sum_horizontal
    ),
    METADATA_FN.VALUES_N_OCCURRENCES: MapReducePair(PRESENT_VALS.len(), pl.sum_horizontal),
    METADATA_FN.VALUES_N_INTS: MapReducePair(VAL.filter(VAL_PRESENT & IS_INT).len(), pl.sum_horizontal),
    METADATA_FN.VALUES_SUM: MapReducePair(PRESENT_VALS.sum(), pl.sum_horizontal),
    METADATA_FN.VALUES_SUM_SQD: MapReducePair((PRESENT_VALS**2).sum(), pl.sum_horizontal),
    METADATA_FN.VALUES_MIN: MapReducePair(PRESENT_VALS.min(), pl.min_horizontal),
    METADATA_FN.VALUES_MAX: MapReducePair(PRESENT_VALS.max(), pl.max_horizontal),
    METADATA_FN.VALUES_QUANTILES: MapReducePair(PRESENT_VALS, quantile_reducer),
}


def validate_args_and_get_code_cols(stage_cfg: DictConfig, code_modifiers: list[str] | None) -> list[str]:
    """Validates the stage configuration and code_modifiers argument and returns the code group keys.

    Args:
        stage_cfg: The configuration object for this stage. It must contain an `aggregations` field that has a
            list of aggregations that should be applied in this stage. Each aggregation must be a string in
            the `METADATA_FN` enumeration.
        code_modifiers: A list of column names that should be used in addition to the core `code`
            column to group the data before applying the aggregations. If None, only the `code` column will be
            used.

    Returns:
        A list of column names that should be used to group the data before applying the aggregations.

    Raises:
        ValueError: If the stage config either does not contain an aggregations field, contains an empty or
            mis-typed aggregations field, or contains an invalid aggregation function.
        ValueError: If the code_modifiers argument is not a list of strings or None.

    Examples:
        >>> no_aggs_cfg = DictConfig({"other_key": "other_value"})
        >>> validate_args_and_get_code_cols(no_aggs_cfg, None)
        Traceback (most recent call last):
            ...
        ValueError: Stage config must contain an 'aggregations' field. Got:
            other_key: other_value
        >>> invalid_agg_cfg = DictConfig({"aggregations": ["INVALID"]})
        >>> validate_args_and_get_code_cols(invalid_agg_cfg, None)
        Traceback (most recent call last):
            ...
        ValueError: Metadata aggregation function INVALID not found in METADATA_FN enumeration. Values are:
            code/n_subjects, code/n_occurrences, values/n_subjects, values/n_occurrences, values/n_ints,
            values/sum, values/sum_sqd, values/min, values/max, values/quantiles
        >>> valid_cfg = DictConfig({"aggregations": ["code/n_subjects", {"name": "values/n_ints"}]})
        >>> validate_args_and_get_code_cols(valid_cfg, 33)
        Traceback (most recent call last):
            ...
        ValueError: code_modifiers must be a list of strings or None. Got 33
        >>> validate_args_and_get_code_cols(valid_cfg, [33])
        Traceback (most recent call last):
            ...
        ValueError: code_modifiers must be a list of strings or None. Got [33]
        >>> validate_args_and_get_code_cols(valid_cfg, ["modifier1"])
        ['code', 'modifier1']
        >>> validate_args_and_get_code_cols(valid_cfg, None)
        ['code']
    """

    if "aggregations" not in stage_cfg:
        raise ValueError(
            f"Stage config must contain an 'aggregations' field. Got:\n{OmegaConf.to_yaml(stage_cfg)}"
        )

    aggregations = stage_cfg.aggregations
    for agg in aggregations:
        if isinstance(agg, (dict, DictConfig)):
            agg = agg.get("name", None)
        if agg not in {fn.value for fn in METADATA_FN}:
            raise ValueError(
                f"Metadata aggregation function {agg} not found in METADATA_FN enumeration. Values are: "
                f"{', '.join([fn.value for fn in METADATA_FN])}"
            )

    match code_modifiers:
        case None:
            return ["code"]
        case list() | ListConfig() if all(isinstance(col, str) for col in code_modifiers):
            return ["code"] + code_modifiers
        case _:
            raise ValueError(f"code_modifiers must be a list of strings or None. Got {code_modifiers}")


def mapper_fntr(
    stage_cfg: DictConfig, code_modifiers: list[str] | None
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Returns a function that extracts code metadata from a MEDS cohort shard.

    Args:
        stage_cfg: The configuration object for this stage. It must contain an `aggregations` field that has a
            list of aggregations that should be applied in this stage. Each aggregation must be a string in
            the `METADATA_FN` enumeration, and the mapper function is specified in the
            `CODE_METADATA_AGGREGATIONS` dictionary.
        code_modifiers: A list of column names that should be used in addition to the core `code`
            column to group the data before applying the aggregations. If None, only the `code` column will be
            used.

    Raises: See `validate_args_and_get_code_cols`.

    Returns:
        A function that extracts the specified metadata from a MEDS cohort shard after grouping by the
        specified code & modifier columns. **Note**: The output of this function will, if
        ``stage_cfg.do_summarize_over_all_codes`` is True, contain the metadata summarizing all observations
        across all codes and subjects in the shard, with both ``code`` and all ``code_modifiers`` set
        to `None` in the output dataframe, in the same format as the code/modifier specific rows with non-null
        values.

    Examples:
        >>> import numpy as np
        >>> df = pl.DataFrame({
        ...     "code":             ["A", "B", "A", "B", "C", "A", "C", "B",          "D"],
        ...     "modifier1":        [1,   2,   1,   2,   1,   2,   1,   2,            None],
        ...     "modifier_ignored": [3,   3,   4,   4,   5,   5,   6,   6,            7],
        ...     "subject_id":       [1,   2,   1,   3,   1,   2,   2,   2,            1],
        ...     "numeric_value":    [1.1, 2.,  1.1, 4.,  5.,  6.,  7.5, float('nan'), None],
        ... })
        >>> df
        shape: (9, 5)
        ┌──────┬───────────┬──────────────────┬────────────┬───────────────┐
        │ code ┆ modifier1 ┆ modifier_ignored ┆ subject_id ┆ numeric_value │
        │ ---  ┆ ---       ┆ ---              ┆ ---        ┆ ---           │
        │ str  ┆ i64       ┆ i64              ┆ i64        ┆ f64           │
        ╞══════╪═══════════╪══════════════════╪════════════╪═══════════════╡
        │ A    ┆ 1         ┆ 3                ┆ 1          ┆ 1.1           │
        │ B    ┆ 2         ┆ 3                ┆ 2          ┆ 2.0           │
        │ A    ┆ 1         ┆ 4                ┆ 1          ┆ 1.1           │
        │ B    ┆ 2         ┆ 4                ┆ 3          ┆ 4.0           │
        │ C    ┆ 1         ┆ 5                ┆ 1          ┆ 5.0           │
        │ A    ┆ 2         ┆ 5                ┆ 2          ┆ 6.0           │
        │ C    ┆ 1         ┆ 6                ┆ 2          ┆ 7.5           │
        │ B    ┆ 2         ┆ 6                ┆ 2          ┆ NaN           │
        │ D    ┆ null      ┆ 7                ┆ 1          ┆ null          │
        └──────┴───────────┴──────────────────┴────────────┴───────────────┘
        >>> stage_cfg = DictConfig({
        ...     "aggregations": ["code/n_subjects", "values/n_ints"],
        ...     "do_summarize_over_all_codes": True
        ... })
        >>> mapper = mapper_fntr(stage_cfg, None)
        >>> mapper(df.lazy()).collect()
        shape: (5, 3)
        ┌──────┬─────────────────┬───────────────┐
        │ code ┆ code/n_subjects ┆ values/n_ints │
        │ ---  ┆ ---             ┆ ---           │
        │ str  ┆ u32             ┆ u32           │
        ╞══════╪═════════════════╪═══════════════╡
        │ null ┆ 3               ┆ 4             │
        │ A    ┆ 2               ┆ 1             │
        │ B    ┆ 2               ┆ 2             │
        │ C    ┆ 2               ┆ 1             │
        │ D    ┆ 1               ┆ 0             │
        └──────┴─────────────────┴───────────────┘
        >>> stage_cfg = DictConfig({"aggregations": ["code/n_subjects", "values/n_ints"]})
        >>> mapper = mapper_fntr(stage_cfg, None)
        >>> mapper(df.lazy()).collect()
        shape: (4, 3)
        ┌──────┬─────────────────┬───────────────┐
        │ code ┆ code/n_subjects ┆ values/n_ints │
        │ ---  ┆ ---             ┆ ---           │
        │ str  ┆ u32             ┆ u32           │
        ╞══════╪═════════════════╪═══════════════╡
        │ A    ┆ 2               ┆ 1             │
        │ B    ┆ 2               ┆ 2             │
        │ C    ┆ 2               ┆ 1             │
        │ D    ┆ 1               ┆ 0             │
        └──────┴─────────────────┴───────────────┘
        >>> code_modifiers = ["modifier1"]
        >>> stage_cfg = DictConfig({"aggregations": ["code/n_subjects", "values/n_ints"]})
        >>> mapper = mapper_fntr(stage_cfg, ListConfig(code_modifiers))
        >>> mapper(df.lazy()).collect()
        shape: (5, 4)
        ┌──────┬───────────┬─────────────────┬───────────────┐
        │ code ┆ modifier1 ┆ code/n_subjects ┆ values/n_ints │
        │ ---  ┆ ---       ┆ ---             ┆ ---           │
        │ str  ┆ i64       ┆ u32             ┆ u32           │
        ╞══════╪═══════════╪═════════════════╪═══════════════╡
        │ A    ┆ 1         ┆ 1               ┆ 0             │
        │ A    ┆ 2         ┆ 1               ┆ 1             │
        │ B    ┆ 2         ┆ 2               ┆ 2             │
        │ C    ┆ 1         ┆ 2               ┆ 1             │
        │ D    ┆ null      ┆ 1               ┆ 0             │
        └──────┴───────────┴─────────────────┴───────────────┘
        >>> stage_cfg = DictConfig({"aggregations": ["code/n_occurrences", "values/sum"]})
        >>> mapper = mapper_fntr(stage_cfg, code_modifiers)
        >>> mapper(df.lazy()).collect()
        shape: (5, 4)
        ┌──────┬───────────┬────────────────────┬────────────┐
        │ code ┆ modifier1 ┆ code/n_occurrences ┆ values/sum │
        │ ---  ┆ ---       ┆ ---                ┆ ---        │
        │ str  ┆ i64       ┆ u32                ┆ f64        │
        ╞══════╪═══════════╪════════════════════╪════════════╡
        │ A    ┆ 1         ┆ 2                  ┆ 2.2        │
        │ A    ┆ 2         ┆ 1                  ┆ 6.0        │
        │ B    ┆ 2         ┆ 3                  ┆ 6.0        │
        │ C    ┆ 1         ┆ 2                  ┆ 12.5       │
        │ D    ┆ null      ┆ 1                  ┆ 0.0        │
        └──────┴───────────┴────────────────────┴────────────┘
        >>> stage_cfg = DictConfig({
        ...     "aggregations": ["code/n_occurrences", "values/sum"],
        ...     "do_summarize_over_all_codes": True,
        ... })
        >>> mapper = mapper_fntr(stage_cfg, code_modifiers)
        >>> mapper(df.lazy()).collect()
        shape: (6, 4)
        ┌──────┬───────────┬────────────────────┬────────────┐
        │ code ┆ modifier1 ┆ code/n_occurrences ┆ values/sum │
        │ ---  ┆ ---       ┆ ---                ┆ ---        │
        │ str  ┆ i64       ┆ u32                ┆ f64        │
        ╞══════╪═══════════╪════════════════════╪════════════╡
        │ null ┆ null      ┆ 9                  ┆ 26.7       │
        │ A    ┆ 1         ┆ 2                  ┆ 2.2        │
        │ A    ┆ 2         ┆ 1                  ┆ 6.0        │
        │ B    ┆ 2         ┆ 3                  ┆ 6.0        │
        │ C    ┆ 1         ┆ 2                  ┆ 12.5       │
        │ D    ┆ null      ┆ 1                  ┆ 0.0        │
        └──────┴───────────┴────────────────────┴────────────┘
        >>> stage_cfg = DictConfig({"aggregations": ["values/n_subjects", "values/n_occurrences"]})
        >>> mapper = mapper_fntr(stage_cfg, code_modifiers)
        >>> mapper(df.lazy()).collect()
        shape: (5, 4)
        ┌──────┬───────────┬───────────────────┬──────────────────────┐
        │ code ┆ modifier1 ┆ values/n_subjects ┆ values/n_occurrences │
        │ ---  ┆ ---       ┆ ---               ┆ ---                  │
        │ str  ┆ i64       ┆ u32               ┆ u32                  │
        ╞══════╪═══════════╪═══════════════════╪══════════════════════╡
        │ A    ┆ 1         ┆ 1                 ┆ 2                    │
        │ A    ┆ 2         ┆ 1                 ┆ 1                    │
        │ B    ┆ 2         ┆ 2                 ┆ 2                    │
        │ C    ┆ 1         ┆ 2                 ┆ 2                    │
        │ D    ┆ null      ┆ 0                 ┆ 0                    │
        └──────┴───────────┴───────────────────┴──────────────────────┘
        >>> stage_cfg = DictConfig({"aggregations": ["values/sum_sqd", "values/min", "values/max"]})
        >>> mapper = mapper_fntr(stage_cfg, code_modifiers)
        >>> mapper(df.lazy()).collect()
        shape: (5, 5)
        ┌──────┬───────────┬────────────────┬────────────┬────────────┐
        │ code ┆ modifier1 ┆ values/sum_sqd ┆ values/min ┆ values/max │
        │ ---  ┆ ---       ┆ ---            ┆ ---        ┆ ---        │
        │ str  ┆ i64       ┆ f64            ┆ f64        ┆ f64        │
        ╞══════╪═══════════╪════════════════╪════════════╪════════════╡
        │ A    ┆ 1         ┆ 2.42           ┆ 1.1        ┆ 1.1        │
        │ A    ┆ 2         ┆ 36.0           ┆ 6.0        ┆ 6.0        │
        │ B    ┆ 2         ┆ 20.0           ┆ 2.0        ┆ 4.0        │
        │ C    ┆ 1         ┆ 81.25          ┆ 5.0        ┆ 7.5        │
        │ D    ┆ null      ┆ 0.0            ┆ null       ┆ null       │
        └──────┴───────────┴────────────────┴────────────┴────────────┘
        >>> stage_cfg = DictConfig({
        ...     "aggregations": [{"name": "values/quantiles", "quantiles": [0.25, 0.5, 0.75]}]
        ... })
        >>> mapper = mapper_fntr(stage_cfg, code_modifiers)
        >>> mapper(df.lazy()).collect().select("code", "modifier1", pl.col("values/quantiles"))
        shape: (5, 3)
        ┌──────┬───────────┬──────────────────┐
        │ code ┆ modifier1 ┆ values/quantiles │
        │ ---  ┆ ---       ┆ ---              │
        │ str  ┆ i64       ┆ list[f64]        │
        ╞══════╪═══════════╪══════════════════╡
        │ A    ┆ 1         ┆ [1.1, 1.1]       │
        │ A    ┆ 2         ┆ [6.0]            │
        │ B    ┆ 2         ┆ [2.0, 4.0]       │
        │ C    ┆ 1         ┆ [5.0, 7.5]       │
        │ D    ┆ null      ┆ []               │
        └──────┴───────────┴──────────────────┘
        >>> stage_cfg = DictConfig({
        ...     "aggregations": [{"name": "values/quantiles", "quantiles": [0.25, 0.5, 0.75]}],
        ...     "do_summarize_over_all_codes": True,
        ... })
        >>> mapper = mapper_fntr(stage_cfg, code_modifiers)
        >>> mapper(df.lazy()).collect().select("code", "modifier1", pl.col("values/quantiles"))
        shape: (6, 3)
        ┌──────┬───────────┬───────────────────┐
        │ code ┆ modifier1 ┆ values/quantiles  │
        │ ---  ┆ ---       ┆ ---               │
        │ str  ┆ i64       ┆ list[f64]         │
        ╞══════╪═══════════╪═══════════════════╡
        │ null ┆ null      ┆ [1.1, 2.0, … 7.5] │
        │ A    ┆ 1         ┆ [1.1, 1.1]        │
        │ A    ┆ 2         ┆ [6.0]             │
        │ B    ┆ 2         ┆ [2.0, 4.0]        │
        │ C    ┆ 1         ┆ [5.0, 7.5]        │
        │ D    ┆ null      ┆ []                │
        └──────┴───────────┴───────────────────┘

    Empty dataframes are handled as you would expect
        >>> df_empty = pl.DataFrame({
        ...     "code": [],
        ...     "modifier1": [],
        ...     "modifier_ignored": [],
        ...     "subject_id": [],
        ...     "numeric_value": [],
        ... }, schema=df.schema)
        >>> stage_cfg = DictConfig({"aggregations": ["values/sum_sqd", "values/min", "values/max"]})
        >>> mapper = mapper_fntr(stage_cfg, code_modifiers)
        >>> mapper(df_empty.lazy()).collect()
        shape: (0, 5)
        ┌──────┬───────────┬────────────────┬────────────┬────────────┐
        │ code ┆ modifier1 ┆ values/sum_sqd ┆ values/min ┆ values/max │
        │ ---  ┆ ---       ┆ ---            ┆ ---        ┆ ---        │
        │ str  ┆ i64       ┆ f64            ┆ f64        ┆ f64        │
        ╞══════╪═══════════╪════════════════╪════════════╪════════════╡
        └──────┴───────────┴────────────────┴────────────┴────────────┘
        >>> stage_cfg = DictConfig({
        ...     "aggregations": ["values/sum_sqd", "values/min", "values/max"],
        ...     "do_summarize_over_all_codes": True,
        ... })
        >>> mapper = mapper_fntr(stage_cfg, code_modifiers)
        >>> mapper(df_empty.lazy()).collect()
        shape: (1, 5)
        ┌──────┬───────────┬────────────────┬────────────┬────────────┐
        │ code ┆ modifier1 ┆ values/sum_sqd ┆ values/min ┆ values/max │
        │ ---  ┆ ---       ┆ ---            ┆ ---        ┆ ---        │
        │ str  ┆ i64       ┆ f64            ┆ f64        ┆ f64        │
        ╞══════╪═══════════╪════════════════╪════════════╪════════════╡
        │ null ┆ null      ┆ 0.0            ┆ null       ┆ null       │
        └──────┴───────────┴────────────────┴────────────┴────────────┘
    """

    code_key_columns = validate_args_and_get_code_cols(stage_cfg, code_modifiers)
    aggregations = stage_cfg.aggregations

    agg_operations = {}
    for agg in aggregations:
        agg_name = agg if isinstance(agg, str) else agg["name"]
        agg_operations[agg_name] = CODE_METADATA_AGGREGATIONS[agg_name].mapper

    def by_code_mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.group_by(code_key_columns).agg(**agg_operations).sort(code_key_columns)

    def all_subjects_mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        local_agg_operations = agg_operations.copy()
        if METADATA_FN.VALUES_QUANTILES in agg_operations:
            local_agg_operations[METADATA_FN.VALUES_QUANTILES] = agg_operations[
                METADATA_FN.VALUES_QUANTILES
            ].implode()
        return df.select(**local_agg_operations)

    if stage_cfg.get("do_summarize_over_all_codes", False):

        def mapper(df: pl.LazyFrame) -> pl.LazyFrame:
            by_code = by_code_mapper(df)
            all_subjects = all_subjects_mapper(df)
            return pl.concat([all_subjects, by_code], how="diagonal_relaxed").select(
                *code_key_columns, *agg_operations.keys()
            )

    else:
        mapper = by_code_mapper

    return mapper


def reducer_fntr(
    stage_cfg: DictConfig, code_modifiers: list[str] | None = None
) -> Callable[[Sequence[pl.DataFrame]], pl.DataFrame]:
    """Returns a function that merges different code metadata files together into an aggregated total.

    Args:
        stage_cfg: The configuration object for this stage. It must contain an `aggregations` field that has a
            list of aggregations that should be applied in this stage. Each aggregation must be a string in
            the `METADATA_FN` enumeration, and the reduction function is specified in the
            `CODE_METADATA_AGGREGATIONS` dictionary.
        code_modifiers: A list of column names that should be used in addition to the core `code`
            column to group the data before applying the aggregations. If None, only the `code` column will be
            used.

    Returns:
        A function that aggregates the specified metadata columns from different extracted metadata shards
        into a total view.

    Raises: See `validate_args_and_get_code_cols`.

    Examples:
        >>> df_1 = pl.DataFrame({
        ...     "code": [None, "A", "A", "B", "C"],
        ...     "modifier1": [None, 1, 2, 1, 2],
        ...     "code/n_subjects":  [10, 1, 1, 2, 2],
        ...     "code/n_occurrences": [13, 2, 1, 3, 2],
        ...     "values/n_subjects":  [8, 1, 1, 2, 2],
        ...     "values/n_occurrences": [12, 2, 1, 3, 2],
        ...     "values/n_ints": [4, 0, 1, 3, 1],
        ...     "values/sum": [13.2, 2.2, 6.0, 14.0, 12.5],
        ...     "values/sum_sqd": [21.3, 2.42, 36.0, 84.0, 81.25],
        ...     "values/min": [-1, 0, -1, 2, 2],
        ...     "values/max": [8.0, 1.1, 6.0, 8.0, 7.5],
        ...     "values/quantiles": [[1.1, 1.1], [6.0], [6.0], [5.0, 7.5], []],
        ... })
        >>> df_2 = pl.DataFrame({
        ...     "code": ["A", "A", "B", "C"],
        ...     "modifier1": [1, 2, 1, None],
        ...     "code/n_subjects":  [3, 3, 4, 4],
        ...     "code/n_occurrences": [10, 11, 8, 11],
        ...     "values/n_subjects":  [0, 1, 2, 2],
        ...     "values/n_occurrences": [0, 4, 3, 2],
        ...     "values/n_ints": [0, 1, 3, 1],
        ...     "values/sum": [0., 7.0, 14.0, 12.5],
        ...     "values/sum_sqd": [0., 103.2, 84.0, 81.25],
        ...     "values/min": [None, -1., 0.2, -2.],
        ...     "values/max": [None, 6.2, 1.0, 1.5],
        ...     "values/quantiles": [[1.3, -1.1, 2.0], [6.0, 1.2], [3.0, 2.5], [11.1, 12.]],
        ... })
        >>> df_3 = pl.DataFrame({
        ...     "code": ["D"],
        ...     "modifier1": [1],
        ...     "code/n_subjects": [2],
        ...     "code/n_occurrences": [2],
        ...     "values/n_subjects": [1],
        ...     "values/n_occurrences": [3],
        ...     "values/n_ints": [3],
        ...     "values/sum": [2],
        ...     "values/sum_sqd": [4],
        ...     "values/min": [0],
        ...     "values/max": [2],
        ...     "values/quantiles": [[]],
        ... })
        >>> df_empty = pl.DataFrame({
        ...     "code": [],
        ...     "modifier1": [],
        ...     "code/n_subjects": [],
        ...     "code/n_occurrences": [],
        ...     "values/n_subjects": [],
        ...     "values/n_occurrences": [],
        ...     "values/n_ints": [],
        ...     "values/sum": [],
        ...     "values/sum_sqd": [],
        ...     "values/min": [],
        ...     "values/max": [],
        ...     "values/quantiles": [],
        ... }, schema=df_3.schema)
        >>> df_null_empty = pl.DataFrame({
        ...     "code": [None],
        ...     "modifier1": [None],
        ...     "code/n_subjects": [0],
        ...     "code/n_occurrences": [0],
        ...     "values/n_subjects": [0],
        ...     "values/n_occurrences": [0],
        ...     "values/n_ints": [0],
        ...     "values/sum": [0],
        ...     "values/sum_sqd": [0],
        ...     "values/min": [None],
        ...     "values/max": [None],
        ...     "values/quantiles": [None],
        ... }, schema=df_3.schema)
        >>> code_modifiers = ["modifier1"]
        >>> stage_cfg = DictConfig({"aggregations": ["code/n_subjects", "values/n_ints"]})
        >>> reducer = reducer_fntr(stage_cfg, code_modifiers)
        >>> reducer(df_1, df_2, df_3, df_empty, df_null_empty)
        shape: (7, 4)
        ┌──────┬───────────┬─────────────────┬───────────────┐
        │ code ┆ modifier1 ┆ code/n_subjects ┆ values/n_ints │
        │ ---  ┆ ---       ┆ ---             ┆ ---           │
        │ str  ┆ i64       ┆ i64             ┆ i64           │
        ╞══════╪═══════════╪═════════════════╪═══════════════╡
        │ null ┆ null      ┆ 10              ┆ 4             │
        │ A    ┆ 1         ┆ 4               ┆ 0             │
        │ A    ┆ 2         ┆ 4               ┆ 2             │
        │ B    ┆ 1         ┆ 6               ┆ 6             │
        │ C    ┆ null      ┆ 4               ┆ 1             │
        │ C    ┆ 2         ┆ 2               ┆ 1             │
        │ D    ┆ 1         ┆ 2               ┆ 3             │
        └──────┴───────────┴─────────────────┴───────────────┘
        >>> cfg = DictConfig({
        ...     "code_modifiers": ["modifier1"],
        ...     "code_processing_stages": {
        ...         "stage1": ["code/n_subjects", "values/n_ints"],
        ...         "stage2": ["code/n_occurrences", "values/sum"],
        ...         "stage3.A": ["values/n_subjects", "values/n_occurrences"],
        ...         "stage3.B": ["values/sum_sqd", "values/min", "values/max"],
        ...         "stage4": ["INVALID"],
        ...     }
        ... })
        >>> stage_cfg = DictConfig({"aggregations": ["code/n_occurrences", "values/sum"]})
        >>> reducer = reducer_fntr(stage_cfg, code_modifiers)
        >>> reducer(df_1, df_2, df_3, df_empty, df_null_empty)
        shape: (7, 4)
        ┌──────┬───────────┬────────────────────┬────────────┐
        │ code ┆ modifier1 ┆ code/n_occurrences ┆ values/sum │
        │ ---  ┆ ---       ┆ ---                ┆ ---        │
        │ str  ┆ i64       ┆ i64                ┆ f64        │
        ╞══════╪═══════════╪════════════════════╪════════════╡
        │ null ┆ null      ┆ 13                 ┆ 13.2       │
        │ A    ┆ 1         ┆ 12                 ┆ 2.2        │
        │ A    ┆ 2         ┆ 12                 ┆ 13.0       │
        │ B    ┆ 1         ┆ 11                 ┆ 28.0       │
        │ C    ┆ null      ┆ 11                 ┆ 12.5       │
        │ C    ┆ 2         ┆ 2                  ┆ 12.5       │
        │ D    ┆ 1         ┆ 2                  ┆ 2.0        │
        └──────┴───────────┴────────────────────┴────────────┘
        >>> stage_cfg = DictConfig({"aggregations": ["values/n_subjects", "values/n_occurrences"]})
        >>> reducer = reducer_fntr(stage_cfg, code_modifiers)
        >>> reducer(df_1, df_2, df_3, df_empty, df_null_empty)
        shape: (7, 4)
        ┌──────┬───────────┬───────────────────┬──────────────────────┐
        │ code ┆ modifier1 ┆ values/n_subjects ┆ values/n_occurrences │
        │ ---  ┆ ---       ┆ ---               ┆ ---                  │
        │ str  ┆ i64       ┆ i64               ┆ i64                  │
        ╞══════╪═══════════╪═══════════════════╪══════════════════════╡
        │ null ┆ null      ┆ 8                 ┆ 12                   │
        │ A    ┆ 1         ┆ 1                 ┆ 2                    │
        │ A    ┆ 2         ┆ 2                 ┆ 5                    │
        │ B    ┆ 1         ┆ 4                 ┆ 6                    │
        │ C    ┆ null      ┆ 2                 ┆ 2                    │
        │ C    ┆ 2         ┆ 2                 ┆ 2                    │
        │ D    ┆ 1         ┆ 1                 ┆ 3                    │
        └──────┴───────────┴───────────────────┴──────────────────────┘
        >>> stage_cfg = DictConfig({"aggregations": ["values/sum_sqd", "values/min", "values/max"]})
        >>> reducer = reducer_fntr(stage_cfg, code_modifiers)
        >>> reducer(df_1, df_2, df_3)
        shape: (7, 5)
        ┌──────┬───────────┬────────────────┬────────────┬────────────┐
        │ code ┆ modifier1 ┆ values/sum_sqd ┆ values/min ┆ values/max │
        │ ---  ┆ ---       ┆ ---            ┆ ---        ┆ ---        │
        │ str  ┆ i64       ┆ f64            ┆ f64        ┆ f64        │
        ╞══════╪═══════════╪════════════════╪════════════╪════════════╡
        │ null ┆ null      ┆ 21.3           ┆ -1.0       ┆ 8.0        │
        │ A    ┆ 1         ┆ 2.42           ┆ 0.0        ┆ 1.1        │
        │ A    ┆ 2         ┆ 139.2          ┆ -1.0       ┆ 6.2        │
        │ B    ┆ 1         ┆ 168.0          ┆ 0.2        ┆ 8.0        │
        │ C    ┆ null      ┆ 81.25          ┆ -2.0       ┆ 1.5        │
        │ C    ┆ 2         ┆ 81.25          ┆ 2.0        ┆ 7.5        │
        │ D    ┆ 1         ┆ 4.0            ┆ 0.0        ┆ 2.0        │
        └──────┴───────────┴────────────────┴────────────┴────────────┘
        >>> reducer(df_1.drop("values/min"), df_2, df_3)
        Traceback (most recent call last):
            ...
        KeyError: 'Column values/min not found in DataFrame 0 for reduction.'
        >>> stage_cfg = DictConfig({
        ...     "aggregations": [{"name": "values/quantiles", "quantiles": [0.25, 0.5, 0.75]}],
        ... })
        >>> reducer = reducer_fntr(stage_cfg, code_modifiers)
        >>> reducer(df_1, df_2, df_3, df_empty, df_null_empty).unnest("values/quantiles")
        shape: (7, 5)
        ┌──────┬───────────┬──────────────────────┬─────────────────────┬──────────────────────┐
        │ code ┆ modifier1 ┆ values/quantile/0.25 ┆ values/quantile/0.5 ┆ values/quantile/0.75 │
        │ ---  ┆ ---       ┆ ---                  ┆ ---                 ┆ ---                  │
        │ str  ┆ i64       ┆ f64                  ┆ f64                 ┆ f64                  │
        ╞══════╪═══════════╪══════════════════════╪═════════════════════╪══════════════════════╡
        │ null ┆ null      ┆ 1.1                  ┆ 1.1                 ┆ 1.1                  │
        │ A    ┆ 1         ┆ 1.3                  ┆ 2.0                 ┆ 2.0                  │
        │ A    ┆ 2         ┆ 6.0                  ┆ 6.0                 ┆ 6.0                  │
        │ B    ┆ 1         ┆ 3.0                  ┆ 5.0                 ┆ 5.0                  │
        │ C    ┆ null      ┆ 11.1                 ┆ 12.0                ┆ 12.0                 │
        │ C    ┆ 2         ┆ null                 ┆ null                ┆ null                 │
        │ D    ┆ 1         ┆ null                 ┆ null                ┆ null                 │
        └──────┴───────────┴──────────────────────┴─────────────────────┴──────────────────────┘
    """

    code_key_columns = validate_args_and_get_code_cols(stage_cfg, code_modifiers)
    aggregations = stage_cfg.aggregations

    agg_operations = {}
    for agg in aggregations:
        if isinstance(agg, (dict, DictConfig)):
            agg_name = agg["name"]
            agg_kwargs = {k: v for k, v in agg.items() if k != "name"}
        else:
            agg_name = agg
            agg_kwargs = {}
        agg_operations[agg_name] = (
            CODE_METADATA_AGGREGATIONS[agg_name]
            .reducer(cs.matches(f"{agg_name}/shard_\\d+"), **agg_kwargs)
            .over(*code_key_columns)
        )

    def reducer(*dfs: Sequence[pl.LazyFrame]) -> pl.LazyFrame:
        renamed_dfs = []
        for i, df in enumerate(dfs):
            agg_selectors = []
            for agg in aggregations:
                if isinstance(agg, (dict, DictConfig)):
                    agg = agg["name"]
                if agg not in df.columns:
                    raise KeyError(f"Column {agg} not found in DataFrame {i} for reduction.")
                agg_selectors.append(pl.col(agg).alias(f"{agg}/shard_{i}"))

            renamed_dfs.append(df.select(*code_key_columns, *agg_selectors))

        df = renamed_dfs[0]
        for rdf in renamed_dfs[1:]:
            df = df.join(rdf, on=code_key_columns, how="full", nulls_equal=True, coalesce=True)

        return df.select(*code_key_columns, **agg_operations).sort(code_key_columns)

    return reducer


def run_map_reduce(cfg: DictConfig):
    """Stored separately so it can be easily imported into the pre-built extraction pipelines."""
    all_out_fps = map_over(cfg, compute_fn=mapper_fntr)

    if cfg.worker != 0:
        logger.info("Code metadata mapping completed. Exiting")
        return

    logger.info("Starting reduction process")

    while not all(is_complete_parquet_file(fp) for fp in all_out_fps):  # pragma: no cover
        logger.info("Waiting to begin reduction for all files to be written...")
        time.sleep(cfg.polling_time)

    start = datetime.now()
    logger.info("All map shards complete! Starting code metadata reduction computation.")
    reducer_fp = Path(cfg.stage_cfg.reducer_output_dir) / "codes.parquet"
    reducer_fp.parent.mkdir(parents=True, exist_ok=True)

    reducer_fn = reducer_fntr(cfg.stage_cfg, cfg.get("code_modifiers", None))
    reduced = reducer_fn(*[pl.scan_parquet(fp, glob=False) for fp in all_out_fps]).with_columns(
        cs.numeric().shrink_dtype().name.keep()
    )

    old_metadata_fp = Path(cfg.stage_cfg.metadata_input_dir) / "codes.parquet"
    join_cols = ["code", *cfg.get("code_modifier_cols", [])]

    if old_metadata_fp.exists():
        logger.info(f"Joining to existing code metadata at {str(old_metadata_fp.resolve())}")
        existing = pl.scan_parquet(old_metadata_fp)
        existing = existing.drop(*[c for c in existing.columns if c in set(reduced.columns) - set(join_cols)])
        reduced = reduced.join(existing, on=join_cols, how="left", coalesce=True)

    write_lazyframe(reduced, reducer_fp)
    logger.info(f"Finished reduction in {datetime.now() - start}")


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Aggregates code metadata across shards.

    Note that the output of this stage includes a `null` code row if
    `stage_configs.STAGE_NAME.do_summarize_over_all_codes` is True. This row contains the total counts across
    all codes, _not_ the counts for rows with a code that is `null`, which should not happen.
    """

    run_map_reduce(cfg)
