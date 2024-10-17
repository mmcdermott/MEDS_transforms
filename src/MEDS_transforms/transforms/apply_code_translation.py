#!/usr/bin/env python
"""TODO."""
from collections.abc import Callable

import hydra
import polars as pl
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def apply_code_translation_fntr(
    stage_cfg: DictConfig, code_metadata: pl.LazyFrame, code_modifiers: list[str] | None = None
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """In addition, the `code_metadata` dataset should contain information about the codes in the MEDS
    dataset,

    including the mandatory columns:
      - `code` (`str`)
      - `code/vocab_index` (`int`)
      - Any `code_modifiers` columns, if specified

    Args:
        df: The MEDS dataset to normalize. See above for the expected schema.
        code_metadata: Metadata about the codes in the MEDS dataset. See above for the expected schema.
        code_modifiers: Additional columns to join on, which will be discarded from the output dataframe.

    Returns:
        The translated MEDS dataset, with the schema described above.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 1, 2, 2, 3, 3],
        ...         "code": ["static", "DOB", "ICD//9//A", "ICD//10//A",
        ...                  "DOB", "ICD//9//A", "ICD//9//B", "ICD//9//C"],
        ...     },
        ...     schema={"subject_id": pl.UInt32, "code": pl.Utf8},
        ... ).lazy()
        >>> stage_cfg = DictConfig({"translation_col": "translated"})
        >>> metadata_df = pl.DataFrame(
        ...     {
        ...         "code": ["static", "DOB", "ICD//9//A", "ICD//9//B", "ICD//9//C", "ICD//10//A"],
        ...         "translated": [None, None, "ICD//10//A", "ICD//10//B", "ICD//10//C", None],
        ...     },
        ...     schema={"code": pl.Utf8, "translated": pl.Utf8},
        ... )
        >>> apply_code_translation_fntr(stage_cfg, metadata_df)(df).collect()
        shape: (8, 2)
        ┌────────────┬────────────┐
        │ subject_id ┆ code       │
        │ ---        ┆ ---        │
        │ u32        ┆ str        │
        ╞════════════╪════════════╡
        │ 1          ┆ static     │
        │ 1          ┆ DOB        │
        │ 1          ┆ ICD//10//A │
        │ 1          ┆ ICD//10//A │
        │ 2          ┆ DOB        │
        │ 2          ┆ ICD//10//A │
        │ 3          ┆ ICD//10//B │
        │ 3          ┆ ICD//10//C │
        └────────────┴────────────┘
        >>> metadata_df = pl.DataFrame(
        ...     {
        ...         "code": ["static", "DOB", "ICD//9//A", "ICD//9//B", "ICD//9//B", "ICD//10//A"],
        ...         "translated": [None, None, "ICD//10//A", "ICD//10//B", "ICD//10//C", None],
        ...     },
        ...     schema={"code": pl.Utf8, "translated": pl.Utf8},
        ... )
        >>> apply_code_translation_fntr(stage_cfg, metadata_df)(df).collect()
        shape: (9, 2)
        ┌────────────┬────────────┐
        │ subject_id ┆ code       │
        │ ---        ┆ ---        │
        │ u32        ┆ str        │
        ╞════════════╪════════════╡
        │ 1          ┆ static     │
        │ 1          ┆ DOB        │
        │ 1          ┆ ICD//10//A │
        │ 1          ┆ ICD//10//A │
        │ 2          ┆ DOB        │
        │ 2          ┆ ICD//10//A │
        │ 3          ┆ ICD//10//B │
        │ 3          ┆ ICD//10//C │
        │ 3          ┆ ICD//9//C  │
        └────────────┴────────────┘
    """
    translation_col = stage_cfg.get("translation_col", None)
    if translation_col is None:
        return lambda df: df

    if translation_col not in code_metadata.columns:
        raise ValueError(f"Column '{translation_col}' not found in code metadata.")

    def apply_code_translation_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.join(code_metadata.select(translation_col, "code").lazy(), on="code", how="left")
            .with_columns(code=pl.coalesce(translation_col, "code"))
            .drop(translation_col)
        )

    return apply_code_translation_fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    map_over(cfg, compute_fn=apply_code_translation_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
