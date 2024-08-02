#!/usr/bin/env python
"""Transformations for normalizing MEDS datasets, across both categorical and continuous dimensions."""

from functools import partial
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def normalize(
    df: pl.LazyFrame, code_metadata: pl.LazyFrame, code_modifiers: list[str] | None = None
) -> pl.LazyFrame:
    """Normalize a MEDS dataset across both categorical and continuous dimensions.

    This function expects a MEDS dataset in flattened form, with columns for:
      - `patient_id`
      - `time`
      - `code`
      - `numerical_value`

    In addition, the `code_metadata` dataset should contain information about the codes in the MEDS dataset,
    including the mandatory columns:
      - `code` (`categorical`)
      - `code/vocab_index` (`int`)
      - Any `code_modifiers` columns, if specified

    Additionally, it must either have:
      - Pre-computed means and standard deviations for the numerical values of the codes in the MEDS dataset,
        via:
        - `values/mean` (`float`)
        - `values/std` (`float`)
      - Or the necessary statistics to compute the per-occurrence mean and standard deviation of the numerical
        values of the codes in the MEDS dataset, via:
        - `values/n_occurrences` (`int`)
        - `values/sum` (`float`)
        - `values/sum_sqd` (`float`)


    The `values/*` functions will be used to normalize the code numerical values to have a mean of 0 and a
    standard deviation of 1. The output dataframe will further be filtered to only contain rows where the
    `code` in the MEDS dataset appears in the `code_metadata` dataset, and the output `code` column will be
    converted to the `code/vocab_index` integral ID from the `code_metadata` dataset.

    This function can further be customized by specifying additional columns to join on, via the
    `code_modifiers` parameter, which must appear in both the MEDS dataset and the code metadata. These
    columns will be discarded from the output dataframe, which will only contain the four expected input
    columns, though normalized.

    Args:
        df: The MEDS dataset to normalize. See above for the expected schema.
        code_metadata: Metadata about the codes in the MEDS dataset. See above for the expected schema.
        code_modifiers: Additional columns to join on, which will be discarded from the output dataframe.

    Returns:
        The normalized MEDS dataset, with the schema described above.

    Examples:
        >>> from datetime import datetime
        >>> MEDS_df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 2, 2, 2, 3],
        ...         "time": [
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...         ],
        ...         "code": ["lab//A", "lab//A", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
        ...         "numerical_value": [1, 3, None, 3, None, None, None],
        ...         "unit": ["mg/dL", "g/dL", None, "mg/dL", None, None, None],
        ...     },
        ...     schema = {
        ...         "patient_id": pl.UInt32,
        ...         "time": pl.Datetime,
        ...         "code": pl.Utf8,
        ...         "numerical_value": pl.Float64,
        ...         "unit": pl.Utf8,
        ...    },
        ... )
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["lab//A", "lab//C", "dx//B", "dx//E", "lab//F"],
        ...         "code/vocab_index": [0, 2, 3, 4, 5],
        ...         "values/mean": [2.0, None, None, None, 3],
        ...         "values/std": [0.5, None, None, None, 0.2],
        ...     },
        ...     schema = {
        ...         "code": pl.Utf8,
        ...         "code/vocab_index": pl.UInt32,
        ...         "values/mean": pl.Float64,
        ...         "values/std": pl.Float64,
        ...     },
        ... )
        >>> normalize(MEDS_df.lazy(), code_metadata.lazy()).collect()
        shape: (6, 4)
        ┌────────────┬─────────────────────┬──────┬─────────────────┐
        │ patient_id ┆ time                ┆ code ┆ numerical_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---             │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64             │
        ╞════════════╪═════════════════════╪══════╪═════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ -2.0            │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ 2.0             │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 3    ┆ null            │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 0    ┆ 2.0             │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 2    ┆ null            │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 5    ┆ null            │
        └────────────┴─────────────────────┴──────┴─────────────────┘
        >>> MEDS_df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 2, 2, 2, 3],
        ...         "time": [
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...             datetime(2022, 10, 2),
        ...         ],
        ...         "code": ["lab//A", "lab//A", "dx//B", "lab//A", "dx//D", "lab//C", "lab//F"],
        ...         "numerical_value": [1, 3, None, 3, None, None, None],
        ...         "unit": ["mg/dL", "g/dL", None, "mg/dL", None, None, None],
        ...     },
        ...     schema = {
        ...         "patient_id": pl.UInt32,
        ...         "time": pl.Datetime,
        ...         "code": pl.Utf8,
        ...         "numerical_value": pl.Float64,
        ...         "unit": pl.Utf8,
        ...    },
        ... )
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["lab//A", "lab//A", "lab//C", "dx//B", "dx//E", "lab//F"],
        ...         "unit": ["mg/dL", "g/dL", None, None, None, None],
        ...         "code/vocab_index": [0, 1, 2, 3, 4, 5],
        ...         "values/mean": [2.0, 3.0, None, None, None, 3],
        ...         "values/std": [0.5, 2.0, None, None, None, 0.2],
        ...     },
        ...     schema = {
        ...         "code": pl.Utf8,
        ...         "unit": pl.Utf8,
        ...         "code/vocab_index": pl.UInt32,
        ...         "values/mean": pl.Float64,
        ...         "values/std": pl.Float64,
        ...     },
        ... )
        >>> normalize(MEDS_df.lazy(), code_metadata.lazy(), ["unit"]).collect()
        shape: (6, 4)
        ┌────────────┬─────────────────────┬──────┬─────────────────┐
        │ patient_id ┆ time                ┆ code ┆ numerical_value │
        │ ---        ┆ ---                 ┆ ---  ┆ ---             │
        │ u32        ┆ datetime[μs]        ┆ u32  ┆ f64             │
        ╞════════════╪═════════════════════╪══════╪═════════════════╡
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 0    ┆ -2.0            │
        │ 1          ┆ 2021-01-01 00:00:00 ┆ 1    ┆ 0.0             │
        │ 1          ┆ 2021-01-02 00:00:00 ┆ 3    ┆ null            │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 0    ┆ 2.0             │
        │ 2          ┆ 2022-10-02 00:00:00 ┆ 2    ┆ null            │
        │ 3          ┆ 2022-10-02 00:00:00 ┆ 5    ┆ null            │
        └────────────┴─────────────────────┴──────┴─────────────────┘
    """

    if code_modifiers is None:
        code_modifiers = []

    cols_to_select = ["code", "code/vocab_index"] + code_modifiers

    mean_col = pl.col("values/sum") / pl.col("values/n_occurrences")
    stddev_col = (pl.col("values/sum_sqd") / pl.col("values/n_occurrences") - mean_col**2) ** 0.5

    code_metadata_columns = set(code_metadata.collect_schema().names())
    if "values/mean" in code_metadata_columns:
        cols_to_select.append("values/mean")
    else:
        cols_to_select.append(mean_col.alias("values/mean"))

    if "values/std" in code_metadata_columns:
        cols_to_select.append("values/std")
    else:
        cols_to_select.append(stddev_col.alias("values/std"))

    idx_col = "_row_idx"
    df_cols = df.collect_schema().names()
    while idx_col in df_cols:
        idx_col = f"_{idx_col}"

    return (
        df.with_row_index(idx_col)
        .join(
            code_metadata.select(cols_to_select),
            on=["code"] + code_modifiers,
            how="inner",
            join_nulls=True,
        )
        .select(
            idx_col,
            "patient_id",
            "time",
            pl.col("code/vocab_index").alias("code"),
            ((pl.col("numerical_value") - pl.col("values/mean")) / pl.col("values/std")).alias(
                "numerical_value"
            ),
        )
        .sort(idx_col)
        .drop(idx_col)
    )


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    code_metadata = pl.read_parquet(
        Path(cfg.stage_cfg.metadata_input_dir) / "codes.parquet", use_pyarrow=True
    ).lazy()
    code_modifiers = cfg.get("code_modifier_columns", None)
    compute_fn = partial(normalize, code_metadata=code_metadata, code_modifiers=code_modifiers)

    map_over(cfg, compute_fn=compute_fn)


if __name__ == "__main__":
    main()
