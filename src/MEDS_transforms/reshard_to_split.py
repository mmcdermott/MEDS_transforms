#!/usr/bin/env python
"""Utilities for re-sharding a MEDS cohort to subsharded splits."""

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import hydra
import polars as pl
import polars.selectors as cs
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.extract.split_and_shard_patients import shard_patients
from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.utils import write_lazyframe


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Re-shard a MEDS cohort to in a manner that subdivides patient splits."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    splits_file = Path(cfg.input_dir) / "metadata" / "patient_splits.parquet"
    splits_df = pl.read_parquet(splits_file, use_pyarrow=True)
    splits_map = defaultdict(list)
    for (pt_id, sp) in splits_df.iterrows():
        splits_map[sp].append(pt_id)


    new_sharded_splits = shard_patients(
        patients = splits_df["patient_id"].to_numpy()
        n_patients_per_shard = cfg.stage_cfg.n_patients_per_shard,
        external_splits = splits_map,
        split_fracs_dict = {},

    external_splits: dict[str, Sequence[SUBJ_ID_T]] | None = None,
    split_fracs_dict: dict[str, float] = {"train": 0.8, "tuning": 0.1, "held_out": 0.1},
    seed: int = 1,



    output_dir = Path(cfg.stage_cfg.output_dir)
    shards_single_output, include_only_train = shard_iterator(cfg)

    if include_only_train:
        raise ValueError("Not supported for this stage.")

    for in_fp, out_fp in shards_single_output:
        sharded_path = out_fp.relative_to(output_dir)

        schema_out_fp = output_dir / "schemas" / sharded_path
        event_seq_out_fp = output_dir / "event_seqs" / sharded_path

        logger.info(f"Tokenizing {str(in_fp.resolve())} into schemas at {str(schema_out_fp.resolve())}")

        rwlock_wrap(
            in_fp,
            schema_out_fp,
            pl.scan_parquet,
            write_lazyframe,
            extract_statics_and_schema,
            do_return=False,
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
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":
    main()
