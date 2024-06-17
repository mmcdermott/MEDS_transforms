#!/usr/bin/env python

import json
import random
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.tokenize import (
    extract_seq_of_patient_events,
    extract_statics_and_schema,
)
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig):
    """TODO."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    input_dir = Path(cfg.stage_cfg.data_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)

    shards = json.loads((Path(cfg.input_dir) / "splits.json").read_text())

    patient_splits = list(shards.keys())
    random.shuffle(patient_splits)

    for sp in patient_splits:
        in_fp = input_dir / f"{sp}.parquet"
        schema_out_fp = output_dir / "schemas" / f"{sp}.parquet"
        event_seq_out_fp = output_dir / "event_seqs" / f"{sp}.parquet"

        logger.info(f"Tokenizing {str(in_fp.resolve())} into schemas at {str(schema_out_fp.resolve())}")

        rwlock_wrap(
            in_fp,
            schema_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_statics_and_schema,
            do_return=False,
            cache_intermediate=False,
            do_overwrite=cfg.do_overwrite,
        )

        logger.info(f"Tokenizing {str(in_fp.resolve())} into event_seqs at {str(event_seq_out_fp.resolve())}")

        rwlock_wrap(
            in_fp,
            event_seq_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_seq_of_patient_events,
            do_return=False,
            cache_intermediate=False,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":
    main()
