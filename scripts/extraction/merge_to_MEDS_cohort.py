#!/usr/bin/env python

import json
import random
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init

pl.enable_string_cache()


def read_fn(sp_dir: Path, unique_by: list[str] | str | None) -> pl.LazyFrame:
    files_to_read = list(sp_dir.glob("**/*.parquet"))

    if not files_to_read:
        raise FileNotFoundError(f"No files found in {sp_dir}/**/*.parquet.")

    file_strs = "\n".join(f"  - {str(fp.resolve())}" for fp in files_to_read)
    logger.info(f"Reading {len(files_to_read)} files:\n{file_strs}")

    dfs = [pl.scan_parquet(fp, glob=False) for fp in files_to_read]
    df = pl.concat(dfs, how="diagonal_relaxed")

    match unique_by:
        case None:
            pass
        case "*":
            df = df.unique(maintain_order=False)
        case list() if len(unique_by) == 0 and all(isinstance(u, str) for u in unique_by):
            subset = []
            for u in unique_by:
                if u in df.columns:
                    subset.append(u)
                else:
                    logger.warning(f"Column {u} not found in dataframe. Omitting from unique-by subset.")
            df = df.unique(maintain_order=False, subset=subset)
        case _:
            raise ValueError(f"Invalid unique_by value: {unique_by}")

    return df.sort(by=["patient_id", "timestamp"], multithreaded=False)


def write_fn(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)


def identity_fn(df: pl.LazyFrame) -> pl.LazyFrame:
    return df


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Merges the patient sub-sharded events into a single parquet file per patient shard."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())

    logger.info("Starting patient shard merging.")

    patient_subsharded_dir = Path(cfg.stage_cfg.data_input_dir)
    if not patient_subsharded_dir.is_dir():
        raise FileNotFoundError(f"Patient sub-sharded directory not found: {patient_subsharded_dir}")

    patient_splits = list(shards.keys())
    random.shuffle(patient_splits)

    reader = partial(read_fn, unique_by=cfg.stage_cfg.get("unique_by", None))

    for sp in patient_splits:
        in_dir = patient_subsharded_dir / sp
        out_fp = Path(cfg.stage_cfg.output_dir) / f"{sp}.parquet"

        shard_fps = sorted(list(in_dir.glob("**/*.parquet")))
        shard_fp_strs = [f"  * {str(fp.resolve())}" for fp in shard_fps]
        logger.info(f"Merging {len(shard_fp_strs)} shards into {out_fp}:\n" + "\n".join(shard_fp_strs))
        rwlock_wrap(in_dir, out_fp, reader, write_fn, identity_fn, do_return=False)

    logger.info("Output cohort written.")


if __name__ == "__main__":
    main()
