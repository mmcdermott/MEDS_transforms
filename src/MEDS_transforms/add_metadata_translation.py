#!/usr/bin/env python
"""TODO."""
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.utils import hydra_loguru_init


def add_metadata_translation(code_metadata: pl.DataFrame, translation_col: str) -> pl.DataFrame:
    """Validate the code metadata has the requisite columns and is unique.

    Args:
        code_metadata: Metadata about the codes in the MEDS dataset, with a column `code` and a collection
            of code modifier columns.
        code_modifiers: The names of the code modifier columns in the `code_metadata` dataset.

    Raises:
        KeyError: If the `code_metadata` dataset does not contain the specified `code_modifiers` or `code`
            columns.
        ValueError: If the `code_metadata` dataset is not unique on the `code` and `code_modifiers` columns.

    Examples:
        >>> code_metadata = pl.DataFrame({
        ...     "code": ["ICD//9//A", "ICD//9//B", "ICD//10//A", "ICD//10//A"],
        ... })
        >>> add_metadata_translation(code_metadata, translation_col="translated")
        shape: (4, 2)
        ┌────────────┬────────────┐
        │ code       ┆ translated │
        │ ---        ┆ ---        │
        │ str        ┆ str        │
        ╞════════════╪════════════╡
        │ ICD//9//A  ┆ null       │
        │ ICD//9//B  ┆ null       │
        │ ICD//10//A ┆ null       │
        │ ICD//10//A ┆ null       │
        └────────────┴────────────┘
    """
    return code_metadata.with_columns(translated=None).with_columns(pl.col.translated.cast(str))


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    output_dir = Path(cfg.stage_cfg.reducer_output_dir)

    code_metadata = pl.read_parquet(metadata_input_dir / "codes.parquet", use_pyarrow=True)

    logger.info("Adding code translation.")

    translation_col = Path(cfg.stage_cfg.translation_col)
    code_metadata = add_metadata_translation(code_metadata, translation_col)

    output_fp = output_dir / "codes.parquet"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Indices assigned. Writing to {output_fp}")

    code_metadata.write_parquet(output_fp, use_pyarrow=True)

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":  # pragma: no cover
    main()
