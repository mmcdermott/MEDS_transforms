#!/usr/bin/env python
"""Utilities for re-sharding a MEDS cohort to subsharded splits."""

import json
import time
from collections import defaultdict
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.extract.split_and_shard_patients import shard_patients
from MEDS_transforms.mapreduce.mapper import identity_fn, read_and_filter_fntr
from MEDS_transforms.mapreduce.utils import (
    is_complete_parquet_file,
    rwlock_wrap,
    shard_iterator,
    shuffle_shards,
)
from MEDS_transforms.utils import stage_init, write_lazyframe


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Re-shard a MEDS cohort to in a manner that subdivides patient splits."""

    stage_init(cfg)

    splits_file = Path(cfg.input_dir) / "metadata" / "patient_splits.parquet"
    splits_df = pl.read_parquet(splits_file, use_pyarrow=True)
    splits_map = defaultdict(list)
    for pt_id, sp in splits_df.iter_rows():
        splits_map[sp].append(pt_id)

    new_sharded_splits = shard_patients(
        patients=splits_df["patient_id"].to_numpy(),
        n_patients_per_shard=cfg.stage_cfg.n_patients_per_shard,
        external_splits=splits_map,
        split_fracs_dict=None,
        seed=cfg.get("seed", 1),
    )

    output_dir = Path(cfg.stage_cfg.output_dir)

    # Write shards to file
    if "shards_map_fp" in cfg:
        shards_fp = Path(cfg.shards_map_fp)
    else:
        shards_fp = output_dir / ".shards.json"

    if shards_fp.is_file():
        if cfg.do_overwrite:
            logger.warning(f"Overwriting {str(shards_fp.resolve())}")
            shards_fp.unlink()
        else:
            old_shards_map = json.loads(shards_fp.read_text())
            if old_shards_map != new_sharded_splits:
                raise FileExistsError(f"{str(shards_fp.resolve())} already exists and shard map differs.")

    shards_fp.write_text(json.dumps(new_sharded_splits))

    output_dir = Path(cfg.stage_cfg.output_dir)

    orig_shards_iter, include_only_train = shard_iterator(cfg, out_suffix="")
    if include_only_train:
        raise ValueError("This stage does not support include_only_train=True")

    orig_shards_iter = [(in_fp, out_fp.relative_to(output_dir)) for in_fp, out_fp in orig_shards_iter]

    new_shards = shuffle_shards(list(new_sharded_splits.keys()), cfg)
    new_shards_iter = [(shard_name, output_dir / shard_name) for shard_name in new_shards]

    # Step 1: Sub-sharding stage
    logger.info("Starting sub-sharding")
    subshard_fps = defaultdict(list)

    for in_fp, orig_shard_name in orig_shards_iter:
        for subshard_name, out_dir in new_shards_iter:
            out_fp = out_dir / f"{orig_shard_name}.parquet"
            subshard_fps[subshard_name].append(out_fp)
            patients = new_sharded_splits[subshard_name]

            if not patients:
                raise ValueError(f"No patients found for {subshard_name}!")

            logger.info(
                f"Sub-sharding {str(in_fp.resolve())} to {len(patients)} patients in {str(out_fp.resolve())}"
            )

            rwlock_wrap(
                in_fp,
                out_fp,
                read_and_filter_fntr(pl.col("patient_id").is_in(patients), pl.scan_parquet),
                write_lazyframe,
                identity_fn,
                do_return=False,
                do_overwrite=cfg.do_overwrite,
            )

    logger.info("Merging sub-shards")
    for subshard_name, subshard_dir in new_shards_iter:
        in_dir = subshard_dir
        in_fps = subshard_fps[subshard_name]
        if not in_fps:
            raise ValueError(f"No subshards found for {subshard_name}!")

        out_fp = subshard_dir.with_suffix(".parquet")

        def read_fn(in_dir: Path) -> pl.LazyFrame:
            while not (all(is_complete_parquet_file(fp) for fp in in_fps) or out_fp.is_file()):
                logger.info("Waiting to begin merging for all sub-shard files to be written...")
                time.sleep(cfg.polling_time)

            logger.info(f"Merging {str(in_dir.resolve())}/**/*.parquet:")
            df = None
            for fp in in_fps:
                logger.info(f"  - {str(fp.resolve())}")
                if df is None:
                    df = pl.scan_parquet(fp, glob=False)
                else:
                    df = df.merge_sorted(pl.scan_parquet(fp, glob=False), key="patient_id")
            return df

        def compute_fn(df: list[pl.DataFrame]) -> pl.LazyFrame:
            logger.info(f"Merging {subshard_dir}/**/*.parquet into {str(out_fp.resolve())}")
            # pl.concat(dfs, how="vertical").lazy()
            return df.sort(by=["patient_id", "time"], maintain_order=True, multithreaded=False)

        def write_fn(df: pl.LazyFrame, out_fp: Path) -> None:
            write_lazyframe(df, out_fp)

            logger.info(f"Cleaning up subsharded files in {str(subshard_dir.resolve())}/*.")
            for fp in in_fps:
                if fp.exists():
                    fp.unlink()
            try:
                for root, dirs, files in subshard_dir.walk(top_down=False):
                    walked_dir = root.relative_to(subshard_dir)
                    if files:
                        raise FileExistsError(f"Files found in {walked_dir} after cleanup!: {files}")
                    for d in dirs:
                        (root / d).rmdir()
                subshard_dir.rmdir()
            except OSError as e:
                contents_str = "\n".join([str(f) for f in subshard_dir.iterdir()])
                raise ValueError(f"Could not remove {str(subshard_dir)}. Contents:\n{contents_str}") from e

        logger.info(f"Merging files to {str(out_fp.resolve())}")
        result_computed, _ = rwlock_wrap(
            in_dir,
            out_fp,
            read_fn,
            write_fn,
            compute_fn,
            do_return=False,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Done with {cfg.stage}")
    return 0


if __name__ == "__main__":
    main()
