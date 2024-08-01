#!/usr/bin/env python
"""A polars-to-polars transformation function for filtering patients by sequence length."""
from collections.abc import Callable
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_measurements_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifier_columns: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that filters patient events to only encompass those with a set of permissible codes.

    Args:
        df: The input DataFrame.
        stage_cfg: The configuration for the code filtering stage.

    Returns:
        The processed DataFrame.

    Examples:
        >>> code_metadata_df = pl.DataFrame({
        ...     "code":               ["A", "A", "B", "C"],
        ...     "modifier1":          [1,   2,   1,   2],
        ...     "code/n_patients":    [2,   1,   3,   2],
        ...     "code/n_occurrences": [4,   5,   3,   2],
        ... })
        >>> data = pl.DataFrame({
        ...     "patient_id": [1,   1,   2,   2],
        ...     "code":       ["A", "B", "A", "C"],
        ...     "modifier1":  [1,   1,   2,   2],
        ... }).lazy()
        >>> stage_cfg = DictConfig({"min_patients_per_code": 2, "min_occurrences_per_code": 3})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (2, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_patients_per_code": 1, "min_occurrences_per_code": 4})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (2, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_patients_per_code": 1})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (4, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        │ 2          ┆ C    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_patients_per_code": None, "min_occurrences_per_code": None})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (4, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 1          ┆ A    ┆ 1         │
        │ 1          ┆ B    ┆ 1         │
        │ 2          ┆ A    ┆ 2         │
        │ 2          ┆ C    ┆ 2         │
        └────────────┴──────┴───────────┘
        >>> stage_cfg = DictConfig({"min_occurrences_per_code": 5})
        >>> fn = filter_measurements_fntr(stage_cfg, code_metadata_df, ["modifier1"])
        >>> fn(data).collect()
        shape: (1, 3)
        ┌────────────┬──────┬───────────┐
        │ patient_id ┆ code ┆ modifier1 │
        │ ---        ┆ ---  ┆ ---       │
        │ i64        ┆ str  ┆ i64       │
        ╞════════════╪══════╪═══════════╡
        │ 2          ┆ A    ┆ 2         │
        └────────────┴──────┴───────────┘
    """

    min_patients_per_code = stage_cfg.get("min_patients_per_code", None)
    min_occurrences_per_code = stage_cfg.get("min_occurrences_per_code", None)

    filter_exprs = []
    if min_patients_per_code is not None:
        filter_exprs.append(pl.col("code/n_patients") >= min_patients_per_code)
    if min_occurrences_per_code is not None:
        filter_exprs.append(pl.col("code/n_occurrences") >= min_occurrences_per_code)

    if not filter_exprs:
        return lambda df: df

    join_cols = ["code"]
    if code_modifier_columns:
        join_cols.extend(code_modifier_columns)

    allowed_code_metadata = (code_metadata.filter(pl.all_horizontal(filter_exprs)).select(join_cols)).lazy()

    def filter_measurements_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        f"""Filters patient events to only encompass those with a set of permissible codes.

        In particular, this function filters the DataFrame to only include (code, modifier) pairs that have
        at least {min_patients_per_code} patients and {min_occurrences_per_code} occurrences.
        """

        idx_col = "_row_idx"
        df_columns = set(df.collect_schema().names())
        while idx_col in df_columns:
            idx_col = f"_{idx_col}"

        return (
            df.with_row_index(idx_col)
            .join(allowed_code_metadata, on=join_cols, how="inner")
            .sort(idx_col)
            .drop(idx_col)
        )

    return filter_measurements_fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    code_metadata = pl.read_parquet(
        Path(cfg.stage_cfg.metadata_input_dir) / "code_metadata.parquet", use_pyarrow=True
    )
    compute_fn = filter_measurements_fntr(cfg.stage_cfg, code_metadata)

    map_over(cfg, compute_fn=compute_fn)


if __name__ == "__main__":
    main()
