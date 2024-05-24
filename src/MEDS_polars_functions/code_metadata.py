"""Utilities for grouping and/or reducing MEDS cohort files by code to collect metadata properties.

TODO: more details
"""

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import NamedTuple

import polars as pl
import polars.selectors as cs
from omegaconf import DictConfig

pl.enable_string_cache()


class METADATA_FN(StrEnum):
    """Enumeration of metadata functions that can be applied to a group of codes.

    This enumeration contains the supported code-metadata collection and aggregation function names that can
    be applied to codes (or, rather, unique code & modifier units) in a MEDS cohort. Each function name is
    mapped, in the below `CODE_METADATA_AGGREGATIONS` dictionary, to mapper and reducer functions that (a)
    collect the raw data at a per code-modifier level from MEDS patient-level shards and (b) aggregates two or
    more per-shard metadata files into a single metadata file, which can be used to merge metadata across all
    shards into a single file.

    Note that, by design, these aggregations are all those that permit simple, single-variable reductions.
    E.g., rather than tracking the mean and standard deviation of a numerical value, we track the sum of the
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
        "code/n_patients": Collects the number of unique patients who have (anywhere in their record) the code
            & modifiers group.
        "code/n_occurrences": Collects the total number of occurrences of the code & modifiers group across
            all observations for all patients.
        "values/n_patients": Collects the number of unique patients who have a non-null, non-nan
            numerical_value field for the code & modifiers group.
        "values/n_occurrences": Collects the total number of non-null, non-nan numerical_value occurrences for
            the code & modifiers group across all observations for all patients.
        "values/n_ints": Collects the number of times the observed, non-null numerical_value for the code &
            modifiers group is an integral value (i.e., a whole number, not an integral type).
        "values/sum": Collects the sum of the non-null, non-nan numerical_value values for the code &
            modifiers group.
        "values/sum_sqd": Collects the sum of the squares of the non-null, non-nan numerical_value values for
            the code
        "values/min": Collects the minimum non-null, non-nan numerical_value value for the code & modifiers
        "values/max": Collects the maximum non-null, non-nan numerical_value value for the code & modifiers
    """

    CODE_N_PATIENTS = "code/n_patients"
    CODE_N_OCCURRENCES = "code/n_occurrences"
    VALUES_N_PATIENTS = "values/n_patients"
    VALUES_N_OCCURRENCES = "values/n_occurrences"
    VALUES_N_INTS = "values/n_ints"
    VALUES_SUM = "values/sum"
    VALUES_SUM_SQD = "values/sum_sqd"
    VALUES_MIN = "values/min"
    VALUES_MAX = "values/max"


class MapReducePair(NamedTuple):
    mapper: pl.Expr
    reducer: Callable[[pl.Expr | Sequence[pl.Expr] | cs._selector_proxy_], pl.Expr]


VAL_PRESENT: pl.Expr = pl.col("numerical_value").is_not_null() & pl.col("numerical_value").is_not_nan()
IS_INT: pl.Expr = pl.col("numerical_value").round() == pl.col("numerical_value")

CODE_METADATA_AGGREGATIONS: dict[METADATA_FN, MapReducePair] = {
    METADATA_FN.CODE_N_PATIENTS: MapReducePair(pl.col("patient_id").n_unique(), pl.sum_horizontal),
    METADATA_FN.CODE_N_OCCURRENCES: MapReducePair(pl.len(), pl.sum_horizontal),
    METADATA_FN.VALUES_N_PATIENTS: MapReducePair(
        pl.col("patient_id").filter(VAL_PRESENT).n_unique(), pl.sum_horizontal
    ),
    METADATA_FN.VALUES_N_OCCURRENCES: MapReducePair(
        pl.col("numerical_value").filter(VAL_PRESENT).len(), pl.sum_horizontal
    ),
    METADATA_FN.VALUES_N_INTS: MapReducePair(
        pl.col("numerical_value").filter(VAL_PRESENT & IS_INT).len(), pl.sum_horizontal
    ),
    METADATA_FN.VALUES_SUM: MapReducePair(
        pl.col("numerical_value").filter(VAL_PRESENT).sum(), pl.sum_horizontal
    ),
    METADATA_FN.VALUES_SUM_SQD: MapReducePair(
        (pl.col("numerical_value").filter(VAL_PRESENT) ** 2).sum(), pl.sum_horizontal
    ),
    METADATA_FN.VALUES_MIN: MapReducePair(
        pl.col("numerical_value").filter(VAL_PRESENT).min(), pl.min_horizontal
    ),
    METADATA_FN.VALUES_MAX: MapReducePair(
        pl.col("numerical_value").filter(VAL_PRESENT).max(), pl.max_horizontal
    ),
}


def mapper_fntr(cfg: DictConfig, stage_name: str) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Returns a function that extracts code metadata from a MEDS cohort shard.

    Args:
        cfg: A pre-processing configuration in `OmegaConf` `DictConfig` format. The configuration should have
            a field `code_processing_stages` that specifies the metadata aggregations to perform for each
            stage. This field should be a dictionary with stage names as keys and lists of metadata
            aggregation functions (from the `METADATA_FN` enumeration) as values.
        stage_name: The name of the stage in the configuration file that specifies the set of metadata
            aggregations to perform in this function.

    Returns:
        A function that extracts the specified metadata from a MEDS cohort shard after grouping by the
        specified code & modifier columns

    Raises:
        KeyError: If the specified stage name is not found in the configuration file.
        KeyError: If any specified aggregation function is not an element of the `METADATA_FN` enumeration.

    Examples:
        >>> cfg = DictConfig({
        ...     "code_modifier_columns": ["modifier1"],
        ...     "code_processing_stages": {
        ...         "stage1": ["code/n_patients", "values/n_ints"],
        ...         "stage2": ["code/n_occurrences", "values/sum"],
        ...         "stage3.A": ["values/n_patients", "values/n_occurrences"],
        ...         "stage3.B": ["values/sum_sqd", "values/min", "values/max"],
        ...         "stage4": ["INVALID"],
        ...     }
        ... })
        >>> import numpy as np
        >>> df = pl.DataFrame({
        ...     "code":   pl.Series(["A", "B", "A", "B", "C", "A", "C", "B",  "D"], dtype=pl.Categorical),
        ...     "modifier1":        [1,   2,   1,   2,   1,   2,   1,   2,    None],
        ...     "modifier_ignored": [3,   3,   4,   4,   5,   5,   6,   6,    7],
        ...     "patient_id":       [1,   2,   1,   3,   1,   2,   2,   2,    1],
        ...     "numerical_value":  [1.1, 2., 1.1, 4.,  5.,  6., 7.5, np.NaN, None],
        ... })
        >>> df
        shape: (9, 5)
        ┌──────┬───────────┬──────────────────┬────────────┬─────────────────┐
        │ code ┆ modifier1 ┆ modifier_ignored ┆ patient_id ┆ numerical_value │
        │ ---  ┆ ---       ┆ ---              ┆ ---        ┆ ---             │
        │ cat  ┆ i64       ┆ i64              ┆ i64        ┆ f64             │
        ╞══════╪═══════════╪══════════════════╪════════════╪═════════════════╡
        │ A    ┆ 1         ┆ 3                ┆ 1          ┆ 1.1             │
        │ B    ┆ 2         ┆ 3                ┆ 2          ┆ 2.0             │
        │ A    ┆ 1         ┆ 4                ┆ 1          ┆ 1.1             │
        │ B    ┆ 2         ┆ 4                ┆ 3          ┆ 4.0             │
        │ C    ┆ 1         ┆ 5                ┆ 1          ┆ 5.0             │
        │ A    ┆ 2         ┆ 5                ┆ 2          ┆ 6.0             │
        │ C    ┆ 1         ┆ 6                ┆ 2          ┆ 7.5             │
        │ B    ┆ 2         ┆ 6                ┆ 2          ┆ NaN             │
        │ D    ┆ null      ┆ 7                ┆ 1          ┆ null            │
        └──────┴───────────┴──────────────────┴────────────┴─────────────────┘
        >>> mapper = mapper_fntr(cfg, "stage1")
        >>> mapper(df.lazy()).collect()
        shape: (5, 4)
        ┌──────┬───────────┬─────────────────┬───────────────┐
        │ code ┆ modifier1 ┆ code/n_patients ┆ values/n_ints │
        │ ---  ┆ ---       ┆ ---             ┆ ---           │
        │ cat  ┆ i64       ┆ u32             ┆ u32           │
        ╞══════╪═══════════╪═════════════════╪═══════════════╡
        │ A    ┆ 1         ┆ 1               ┆ 0             │
        │ A    ┆ 2         ┆ 1               ┆ 1             │
        │ B    ┆ 2         ┆ 2               ┆ 2             │
        │ C    ┆ 1         ┆ 2               ┆ 1             │
        │ D    ┆ null      ┆ 1               ┆ 0             │
        └──────┴───────────┴─────────────────┴───────────────┘
        >>> mapper = mapper_fntr(cfg, "stage2")
        >>> mapper(df.lazy()).collect()
        shape: (5, 4)
        ┌──────┬───────────┬────────────────────┬────────────┐
        │ code ┆ modifier1 ┆ code/n_occurrences ┆ values/sum │
        │ ---  ┆ ---       ┆ ---                ┆ ---        │
        │ cat  ┆ i64       ┆ u32                ┆ f64        │
        ╞══════╪═══════════╪════════════════════╪════════════╡
        │ A    ┆ 1         ┆ 2                  ┆ 2.2        │
        │ A    ┆ 2         ┆ 1                  ┆ 6.0        │
        │ B    ┆ 2         ┆ 3                  ┆ 6.0        │
        │ C    ┆ 1         ┆ 2                  ┆ 12.5       │
        │ D    ┆ null      ┆ 1                  ┆ 0.0        │
        └──────┴───────────┴────────────────────┴────────────┘
        >>> mapper = mapper_fntr(cfg, "stage3.A")
        >>> mapper(df.lazy()).collect()
        shape: (5, 4)
        ┌──────┬───────────┬───────────────────┬──────────────────────┐
        │ code ┆ modifier1 ┆ values/n_patients ┆ values/n_occurrences │
        │ ---  ┆ ---       ┆ ---               ┆ ---                  │
        │ cat  ┆ i64       ┆ u32               ┆ u32                  │
        ╞══════╪═══════════╪═══════════════════╪══════════════════════╡
        │ A    ┆ 1         ┆ 1                 ┆ 2                    │
        │ A    ┆ 2         ┆ 1                 ┆ 1                    │
        │ B    ┆ 2         ┆ 2                 ┆ 2                    │
        │ C    ┆ 1         ┆ 2                 ┆ 2                    │
        │ D    ┆ null      ┆ 0                 ┆ 0                    │
        └──────┴───────────┴───────────────────┴──────────────────────┘
        >>> mapper = mapper_fntr(cfg, "stage3.B")
        >>> mapper(df.lazy()).collect()
        shape: (5, 5)
        ┌──────┬───────────┬────────────────┬────────────┬────────────┐
        │ code ┆ modifier1 ┆ values/sum_sqd ┆ values/min ┆ values/max │
        │ ---  ┆ ---       ┆ ---            ┆ ---        ┆ ---        │
        │ cat  ┆ i64       ┆ f64            ┆ f64        ┆ f64        │
        ╞══════╪═══════════╪════════════════╪════════════╪════════════╡
        │ A    ┆ 1         ┆ 2.42           ┆ 1.1        ┆ 1.1        │
        │ A    ┆ 2         ┆ 36.0           ┆ 6.0        ┆ 6.0        │
        │ B    ┆ 2         ┆ 20.0           ┆ 2.0        ┆ 4.0        │
        │ C    ┆ 1         ┆ 81.25          ┆ 5.0        ┆ 7.5        │
        │ D    ┆ null      ┆ 0.0            ┆ null       ┆ null       │
        └──────┴───────────┴────────────────┴────────────┴────────────┘
        >>> mapper = mapper_fntr(cfg, "stage4") # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        KeyError: 'Metadata aggregation function INVALID not found in METADATA_FN enumeration. Values are:
            code/n_patients, code/n_occurrences, values/n_patients, values/n_occurrences, values/n_ints,
            values/sum, values/sum_sqd, values/min, values/max'
        >>> mapper = mapper_fntr(cfg, "stage5")
        Traceback (most recent call last):
            ...
        KeyError: 'Stage name stage5 not found in code_processing_stages in configuration file.'
    """

    if stage_name not in cfg.code_processing_stages:
        raise KeyError(f"Stage name {stage_name} not found in code_processing_stages in configuration file.")

    aggregations = cfg.code_processing_stages[stage_name]
    for agg in aggregations:
        if agg not in METADATA_FN:
            raise KeyError(
                f"Metadata aggregation function {agg} not found in METADATA_FN enumeration. Values are: "
                f"{', '.join([fn.value for fn in METADATA_FN])}"
            )

    code_key_columns = ["code"] + cfg.get("code_modifier_columns", [])
    agg_operations = {agg: CODE_METADATA_AGGREGATIONS[agg].mapper for agg in aggregations}

    def mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.groupby(code_key_columns).agg(**agg_operations).sort(code_key_columns)

    return mapper


def reducer_fntr(cfg: DictConfig, stage_name: str) -> Callable[[Sequence[pl.DataFrame]], pl.DataFrame]:
    """Returns a function that merges different code metadata files together into an aggregated total.

    The functions specified are determined by the TODO field in the configuration file at stage `stage_name`.
    The reductions for these aggregation functions are specified in the `CODE_METADATA_AGGREGATIONS`
    dictionary.

    Args:
        cfg: A pre-processing configuration in `OmegaConf` `DictConfig` format. The configuration should have
            a field `code_processing_stages` that specifies the metadata aggregations to perform for each
            stage. This field should be a dictionary with stage names as keys and lists of metadata
            aggregation functions (from the `METADATA_FN` enumeration) as values.
        stage_name: The name of the stage in the configuration file that specifies the set of metadata
            aggregations to perform in this function.

    Returns:
        A function that aggregates the specified metadata columns from different extracted metadata shards
        into a total view.

    Raises:
        KeyError: If the specified stage name is not found in the configuration file.
        KeyError: If any specified aggregation function is not an element of the `METADATA_FN` enumeration.

    Examples:
        >>> cfg = DictConfig({
        ...     "code_modifier_columns": ["modifier1"],
        ...     "code_processing_stages": {
        ...         "stage1": ["code/n_patients", "values/n_ints"],
        ...         "stage2": ["code/n_occurrences", "values/sum"],
        ...         "stage3.A": ["values/n_patients", "values/n_occurrences"],
        ...         "stage3.B": ["values/sum_sqd", "values/min", "values/max"],
        ...         "stage4": ["INVALID"],
        ...     }
        ... })
        >>> df_1 = pl.DataFrame({
        ...     "code":      pl.Series(["A", "A", "B", "C"], dtype=pl.Categorical),
        ...     "modifier1": [1,   2,   1,   2],
        ...     "code/n_patients":  [1, 1, 2, 2],
        ...     "code/n_occurrences": [2, 1, 3, 2],
        ...     "values/n_patients":  [1, 1, 2, 2],
        ...     "values/n_occurrences": [2, 1, 3, 2],
        ...     "values/n_ints": [0, 1, 3, 1],
        ...     "values/sum": [2.2, 6.0, 14.0, 12.5],
        ...     "values/sum_sqd": [2.42, 36.0, 84.0, 81.25],
        ...     "values/min": [0, -1, 2, 2.],
        ...     "values/max": [1.1, 6.0, 8.0, 7.5],
        ... })
        >>> df_2 = pl.DataFrame({
        ...     "code": pl.Series(["A", "A", "B", "C"], dtype=pl.Categorical),
        ...     "modifier1": [1,   2,   1,   None],
        ...     "code/n_patients":  [3, 3, 4, 4],
        ...     "code/n_occurrences": [10, 11, 8, 11],
        ...     "values/n_patients":  [0, 1, 2, 2],
        ...     "values/n_occurrences": [0, 4, 3, 2],
        ...     "values/n_ints": [0, 1, 3, 1],
        ...     "values/sum": [0., 7.0, 14.0, 12.5],
        ...     "values/sum_sqd": [0., 103.2, 84.0, 81.25],
        ...     "values/min": [None, -1, 0.2, -2.],
        ...     "values/max": [None, 6.2, 1.0, 1.5],
        ... })
        >>> df_3 = pl.DataFrame({
        ...     "code":      pl.Series(["D"], dtype=pl.Categorical),
        ...     "modifier1": [1],
        ...     "code/n_patients":  [2],
        ...     "code/n_occurrences": [2],
        ...     "values/n_patients":  [1],
        ...     "values/n_occurrences": [3],
        ...     "values/n_ints": [3],
        ...     "values/sum": [2],
        ...     "values/sum_sqd": [4],
        ...     "values/min": [0],
        ...     "values/max": [2],
        ... })
        >>> reducer = reducer_fntr(cfg, "stage1")
        >>> reducer(df_1, df_2, df_3)
        shape: (6, 4)
        ┌──────┬───────────┬─────────────────┬───────────────┐
        │ code ┆ modifier1 ┆ code/n_patients ┆ values/n_ints │
        │ ---  ┆ ---       ┆ ---             ┆ ---           │
        │ cat  ┆ i64       ┆ i64             ┆ i64           │
        ╞══════╪═══════════╪═════════════════╪═══════════════╡
        │ A    ┆ 1         ┆ 4               ┆ 0             │
        │ A    ┆ 2         ┆ 4               ┆ 2             │
        │ B    ┆ 1         ┆ 6               ┆ 6             │
        │ C    ┆ null      ┆ 4               ┆ 1             │
        │ C    ┆ 2         ┆ 2               ┆ 1             │
        │ D    ┆ 1         ┆ 2               ┆ 3             │
        └──────┴───────────┴─────────────────┴───────────────┘
        >>> reducer = reducer_fntr(cfg, "stage2")
        >>> reducer(df_1, df_2, df_3)
        shape: (6, 4)
        ┌──────┬───────────┬────────────────────┬────────────┐
        │ code ┆ modifier1 ┆ code/n_occurrences ┆ values/sum │
        │ ---  ┆ ---       ┆ ---                ┆ ---        │
        │ cat  ┆ i64       ┆ i64                ┆ f64        │
        ╞══════╪═══════════╪════════════════════╪════════════╡
        │ A    ┆ 1         ┆ 12                 ┆ 2.2        │
        │ A    ┆ 2         ┆ 12                 ┆ 13.0       │
        │ B    ┆ 1         ┆ 11                 ┆ 28.0       │
        │ C    ┆ null      ┆ 11                 ┆ 12.5       │
        │ C    ┆ 2         ┆ 2                  ┆ 12.5       │
        │ D    ┆ 1         ┆ 2                  ┆ 2.0        │
        └──────┴───────────┴────────────────────┴────────────┘
        >>> reducer = reducer_fntr(cfg, "stage3.A")
        >>> reducer(df_1, df_2, df_3)
        shape: (6, 4)
        ┌──────┬───────────┬───────────────────┬──────────────────────┐
        │ code ┆ modifier1 ┆ values/n_patients ┆ values/n_occurrences │
        │ ---  ┆ ---       ┆ ---               ┆ ---                  │
        │ cat  ┆ i64       ┆ i64               ┆ i64                  │
        ╞══════╪═══════════╪═══════════════════╪══════════════════════╡
        │ A    ┆ 1         ┆ 1                 ┆ 2                    │
        │ A    ┆ 2         ┆ 2                 ┆ 5                    │
        │ B    ┆ 1         ┆ 4                 ┆ 6                    │
        │ C    ┆ null      ┆ 2                 ┆ 2                    │
        │ C    ┆ 2         ┆ 2                 ┆ 2                    │
        │ D    ┆ 1         ┆ 1                 ┆ 3                    │
        └──────┴───────────┴───────────────────┴──────────────────────┘
        >>> reducer = reducer_fntr(cfg, "stage3.B")
        >>> reducer(df_1, df_2, df_3)
        shape: (6, 5)
        ┌──────┬───────────┬────────────────┬────────────┬────────────┐
        │ code ┆ modifier1 ┆ values/sum_sqd ┆ values/min ┆ values/max │
        │ ---  ┆ ---       ┆ ---            ┆ ---        ┆ ---        │
        │ cat  ┆ i64       ┆ f64            ┆ f64        ┆ f64        │
        ╞══════╪═══════════╪════════════════╪════════════╪════════════╡
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
        >>> reducer = reducer_fntr(cfg, "stage4") # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        KeyError: 'Metadata aggregation function INVALID not found in METADATA_FN enumeration. Values are:
            code/n_patients, code/n_occurrences, values/n_patients, values/n_occurrences, values/n_ints,
            values/sum, values/sum_sqd, values/min, values/max'
        >>> reducer = reducer_fntr(cfg, "stage5")
        Traceback (most recent call last):
            ...
        KeyError: 'Stage name stage5 not found in code_processing_stages in configuration file.'
    """

    if stage_name not in cfg.code_processing_stages:
        raise KeyError(f"Stage name {stage_name} not found in code_processing_stages in configuration file.")

    aggregations = cfg.code_processing_stages[stage_name]
    for agg in aggregations:
        if agg not in METADATA_FN:
            raise KeyError(
                f"Metadata aggregation function {agg} not found in METADATA_FN enumeration. Values are: "
                f"{', '.join([fn.value for fn in METADATA_FN])}"
            )

    code_key_columns = ["code"] + cfg.get("code_modifier_columns", [])
    agg_operations = {
        agg: CODE_METADATA_AGGREGATIONS[agg].reducer(cs.matches(f"{agg}/shard_\\d+")) for agg in aggregations
    }

    def reducer(*dfs: Sequence[pl.LazyFrame]) -> pl.LazyFrame:
        renamed_dfs = []
        for i, df in enumerate(dfs):
            for agg in aggregations:
                if agg not in df.columns:
                    raise KeyError(f"Column {agg} not found in DataFrame {i} for reduction.")

            renamed_dfs.append(
                df.select(*code_key_columns, *[pl.col(agg).alias(f"{agg}/shard_{i}") for agg in aggregations])
            )

        df = renamed_dfs[0]
        for rdf in renamed_dfs[1:]:
            df = df.join(rdf, on=code_key_columns, how="full", join_nulls=True, coalesce=True)

        return df.select(*code_key_columns, **agg_operations).sort(code_key_columns)

    return reducer
