"""Utilities for grouping and/or reducing MEDS cohort files by code to collect metadata properties.

TODO: more details
"""

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import NamedTuple

import polars as pl
import polars.selectors as cs
from omegaconf import DictConfig


class METADATA_FN(StrEnum):
    """Enumeration of metadata functions that can be applied to a group of codes.

    This enumeration contains the supported code-metadata collection and aggregation function names that can
    be applied to codes (or, rather, unique code & modifier units) in a MEDS cohort. Each function name is
    mapped, in the below `CODE_METADATA_AGGREGATIONS` dictionary, to mapper and reducer functions that (a)
    collect the raw data at a per code-modifier level from MEDS patient-level shards and (b) aggregates two or
    more per-shard metadata files into a single metadata file, which can be used to merge metadata across all
    shards into a single file.

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
        "values/n_unique": Collects the number of unique, non-null numerical_value values observed for the
            code & modifiers group. NaNs are not counted as unique values.
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
    VALUES_N_UNIQUE = "values/n_unique"
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
    METADATA_FN.VALUES_N_UNIQUE: MapReducePair(
        pl.col("numerical_value").filter(VAL_PRESENT).n_unique(), pl.sum_horizontal
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
        ...         "stage3.A": ["values/n_patients", "values/n_occurrences", "values/n_unique"],
        ...         "stage3.B": ["values/sum_sqd", "values/min", "values/max"],
        ...         "stage4": ["INVALID"],
        ...     }
        ... })
        >>> df = pl.DataFrame({
        ...     "code":             ["A", "B", "A", "B", "C", "A", "C", "B"],
        ...     "modifier1":        [1,   2,   1,   2,   1,   2,   1,   2],
        ...     "modifier_ignored": [3,   3,   4,   4,   5,   5,   6,   6],
        ...     "patient_id":       [1,   2,   1,   3,   1,   2,   2,   2],
        ...     "numerical_value":  [1.1, 2.0, 1.1, 4.0, 5.0, 6.0, 7.5, 8.0],
        ... })
        >>> df
        shape: (8, 5)
        ┌──────┬───────────┬──────────────────┬────────────┬─────────────────┐
        │ code ┆ modifier1 ┆ modifier_ignored ┆ patient_id ┆ numerical_value │
        │ ---  ┆ ---       ┆ ---              ┆ ---        ┆ ---             │
        │ str  ┆ i64       ┆ i64              ┆ i64        ┆ f64             │
        ╞══════╪═══════════╪══════════════════╪════════════╪═════════════════╡
        │ A    ┆ 1         ┆ 3                ┆ 1          ┆ 1.1             │
        │ B    ┆ 2         ┆ 3                ┆ 2          ┆ 2.0             │
        │ A    ┆ 1         ┆ 4                ┆ 1          ┆ 1.1             │
        │ B    ┆ 2         ┆ 4                ┆ 3          ┆ 4.0             │
        │ C    ┆ 1         ┆ 5                ┆ 1          ┆ 5.0             │
        │ A    ┆ 2         ┆ 5                ┆ 2          ┆ 6.0             │
        │ C    ┆ 1         ┆ 6                ┆ 2          ┆ 7.5             │
        │ B    ┆ 2         ┆ 6                ┆ 2          ┆ 8.0             │
        └──────┴───────────┴──────────────────┴────────────┴─────────────────┘
        >>> mapper = mapper_fntr(cfg, "stage1")
        >>> mapper(df)
        shape: (4, 4)
        ┌──────┬───────────┬─────────────────┬───────────────┐
        │ code ┆ modifier1 ┆ code/n_patients ┆ values/n_ints │
        │ ---  ┆ ---       ┆ ---             ┆ ---           │
        │ str  ┆ i64       ┆ u32             ┆ u32           │
        ╞══════╪═══════════╪═════════════════╪═══════════════╡
        │ A    ┆ 1         ┆ 1               ┆ 0             │
        │ A    ┆ 2         ┆ 1               ┆ 1             │
        │ B    ┆ 2         ┆ 2               ┆ 3             │
        │ C    ┆ 1         ┆ 2               ┆ 1             │
        └──────┴───────────┴─────────────────┴───────────────┘
        >>> mapper = mapper_fntr(cfg, "stage2")
        >>> mapper(df)
        shape: (4, 4)
        ┌──────┬───────────┬────────────────────┬────────────┐
        │ code ┆ modifier1 ┆ code/n_occurrences ┆ values/sum │
        │ ---  ┆ ---       ┆ ---                ┆ ---        │
        │ str  ┆ i64       ┆ u32                ┆ f64        │
        ╞══════╪═══════════╪════════════════════╪════════════╡
        │ A    ┆ 1         ┆ 2                  ┆ 2.2        │
        │ A    ┆ 2         ┆ 1                  ┆ 6.0        │
        │ B    ┆ 2         ┆ 3                  ┆ 14.0       │
        │ C    ┆ 1         ┆ 2                  ┆ 12.5       │
        └──────┴───────────┴────────────────────┴────────────┘
        >>> mapper = mapper_fntr(cfg, "stage3.A")
        >>> mapper(df)
        shape: (4, 5)
        ┌──────┬───────────┬───────────────────┬──────────────────────┬─────────────────┐
        │ code ┆ modifier1 ┆ values/n_patients ┆ values/n_occurrences ┆ values/n_unique │
        │ ---  ┆ ---       ┆ ---               ┆ ---                  ┆ ---             │
        │ str  ┆ i64       ┆ u32               ┆ u32                  ┆ u32             │
        ╞══════╪═══════════╪═══════════════════╪══════════════════════╪═════════════════╡
        │ A    ┆ 1         ┆ 1                 ┆ 2                    ┆ 1               │
        │ A    ┆ 2         ┆ 1                 ┆ 1                    ┆ 1               │
        │ B    ┆ 2         ┆ 2                 ┆ 3                    ┆ 3               │
        │ C    ┆ 1         ┆ 2                 ┆ 2                    ┆ 2               │
        └──────┴───────────┴───────────────────┴──────────────────────┴─────────────────┘
        >>> mapper = mapper_fntr(cfg, "stage3.B")
        >>> mapper(df)
        shape: (4, 5)
        ┌──────┬───────────┬────────────────┬────────────┬────────────┐
        │ code ┆ modifier1 ┆ values/sum_sqd ┆ values/min ┆ values/max │
        │ ---  ┆ ---       ┆ ---            ┆ ---        ┆ ---        │
        │ str  ┆ i64       ┆ f64            ┆ f64        ┆ f64        │
        ╞══════╪═══════════╪════════════════╪════════════╪════════════╡
        │ A    ┆ 1         ┆ 2.42           ┆ 1.1        ┆ 1.1        │
        │ A    ┆ 2         ┆ 36.0           ┆ 6.0        ┆ 6.0        │
        │ B    ┆ 2         ┆ 84.0           ┆ 2.0        ┆ 8.0        │
        │ C    ┆ 1         ┆ 81.25          ┆ 5.0        ┆ 7.5        │
        └──────┴───────────┴────────────────┴────────────┴────────────┘
        >>> mapper = mapper_fntr(cfg, "stage4") # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        KeyError: 'Metadata aggregation function INVALID not found in METADATA_FN enumeration. Values are:
            code/n_patients, code/n_occurrences, values/n_patients, values/n_occurrences, values/n_unique,
            values/n_ints, values/sum, values/sum_sqd, values/min, values/max'
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

    def mapper(df: pl.DataFrame) -> pl.DataFrame:
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

    Examples: TODO
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

    def reducer(*dfs: Sequence[pl.DataFrame]) -> pl.DataFrame:
        renamed_dfs = []
        for i, df in enumerate(dfs):
            for agg in aggregations:
                if agg not in df.columns:
                    raise KeyError(f"Column {agg} not found in DataFrame {i} for reduction.")

            dfs.append(df.select(*code_key_columns, *[f"{agg}/shard_{i}" for agg in aggregations]))

        df = renamed_dfs[0]
        for rdf in renamed_dfs[1:]:
            df = df.join(rdf, on=code_key_columns, how="outer")

        return df.select(*code_key_columns, **agg_operations)

    return reducer
