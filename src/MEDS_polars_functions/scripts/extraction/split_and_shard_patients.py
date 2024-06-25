#!/usr/bin/env python
import json
from importlib.resources import files
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.sharding import shard_patients
from MEDS_polars_functions.utils import hydra_loguru_init

config_yaml = files("MEDS_polars_functions").joinpath("configs/extraction.yaml")


@hydra.main(version_base=None, config_path=str(config_yaml.parent), config_name=config_yaml.stem)
def main(cfg: DictConfig):
    """Extracts the set of unique patients from the raw data and splits/shards them and saves the result."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    logger.info("Starting patient splitting and sharding")

    MEDS_cohort_dir = Path(cfg.stage_cfg.output_dir)
    subsharded_dir = Path(cfg.stage_cfg.data_input_dir)

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info(
        f"Reading event conversion config from {event_conversion_cfg_fp} (needed for patient ID columns)"
    )
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    dfs = []

    default_patient_id_col = event_conversion_cfg.pop("patient_id_col", "patient_id")
    for input_prefix, event_cfgs in event_conversion_cfg.items():
        input_patient_id_column = event_cfgs.get("patient_id_col", default_patient_id_col)

        input_fps = list((subsharded_dir / input_prefix).glob("**/*.parquet"))

        input_fps_strs = "\n".join(f"  - {str(fp.resolve())}" for fp in input_fps)
        logger.info(f"Reading patient IDs from {input_prefix} files:\n{input_fps_strs}")

        for input_fp in input_fps:
            dfs.append(
                pl.scan_parquet(input_fp, glob=False)
                .select(pl.col(input_patient_id_column).alias("patient_id"))
                .unique()
            )

    logger.info(f"Joining all patient IDs from {len(dfs)} dataframes")
    patient_ids = (
        pl.concat(dfs)
        .select(pl.col("patient_id").unique())
        .collect(streaming=True)["patient_id"]
        .to_numpy(use_pyarrow=True)
    )

    logger.info(f"Found {len(patient_ids)} unique patient IDs of type {patient_ids.dtype}")

    if cfg.stage_cfg.external_splits_json_fp:
        external_splits_json_fp = Path(cfg.stage_cfg.external_splits_json_fp)
        if not external_splits_json_fp.exists():
            raise FileNotFoundError(f"External splits JSON file not found at {external_splits_json_fp}")

        logger.info(f"Reading external splits from {cfg.external_splits_json_fp}")
        external_splits = json.loads(external_splits_json_fp.read_text())

        size_strs = ", ".join(f"{k}: {len(v)}" for k, v in external_splits.items())
        logger.info(f"Loaded external splits of size: {size_strs}")
    else:
        external_splits = None

    logger.info("Sharding and splitting patients")

    sharded_patients = shard_patients(
        patients=patient_ids,
        external_splits=external_splits,
        split_fracs_dict=cfg.stage_cfg.split_fracs,
        n_patients_per_shard=cfg.stage_cfg.n_patients_per_shard,
        seed=cfg.seed,
    )

    logger.info(f"Writing sharded patients to {MEDS_cohort_dir}")
    MEDS_cohort_dir.mkdir(parents=True, exist_ok=True)
    out_fp = MEDS_cohort_dir / "splits.json"
    out_fp.write_text(json.dumps(sharded_patients))
    logger.info(f"Done writing sharded patients to {out_fp}")


if __name__ == "__main__":
    main()
