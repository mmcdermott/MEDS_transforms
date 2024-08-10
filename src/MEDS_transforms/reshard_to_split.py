#!/usr/bin/env python
"""Utilities for re-sharding a MEDS cohort to subsharded splits."""

import json
from collections import defaultdict
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.extract.split_and_shard_patients import shard_iterator_by_shard_map, shard_patients
from MEDS_transforms.mapreduce.mapper import identity_fn, read_and_filter_fntr
from MEDS_transforms.mapreduce.utils import rwlock_wrap, shard_iterator
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
    for pt_id, sp in splits_df.iterrows():
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
            raise FileExistsError(f"{str(shards_fp.resolve())} already exists.")

    shards_fp.write_text(json.dumps(new_sharded_splits))

    data_input_dir = Path(cfg.stage_cfg.data_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)

    orig_shards_iter, include_only_train = shard_iterator(cfg, out_suffix="")
    if include_only_train:
        raise ValueError("This stage does not support include_only_train=True")

    orig_shards_iter = [(in_fp, out_fp.relative_to(output_dir)) for in_fp, out_fp in orig_shards_iter]

    # Here, we modify the config in a hacky way to make the right shard mapping
    cfg.shards_map_fp = str(shards_fp.resolve())
    cfg.stage_cfg.data_input_dir = str(output_dir.resolve())

    new_shards_iter, include_only_train = shard_iterator_by_shard_map(cfg)
    if include_only_train:
        raise ValueError("This stage does not support include_only_train=True")

    cfg.stage_cfg.data_input_dir = str(data_input_dir.resolve())
    new_shards_iter = [
        (in_fp.relative_to(data_input_dir).with_suffix(""), out_fp.with_suffix(""))
        for in_fp, out_fp in new_shards_iter
    ]

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

            logger.info(f"Sub-sharding {str(in_fp.resolve())} into {str(out_fp.resolve())}")

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
        out_fp = subshard_dir.with_suffix(".parquet")
        if out_fp.is_file():
            if cfg.do_overwrite:
                logger.warning(f"Overwriting {str(out_fp.resolve())}")
                out_fp.unlink()
            else:
                raise FileExistsError(f"{str(out_fp.resolve())} already exists.")

        logger.info(f"Merging {subshard_dir}/**/*.parquet into {str(out_fp.resolve())}")

        if not subshard_fps:
            raise ValueError(f"No subshards found for {subshard_name}!")

        in_dir = subshard_dir
        in_fps = subshard_fps[subshard_name]

        def read_fn(fp: Path) -> pl.LazyFrame:
            return pl.concat([pl.scan_parquet(fp, glob=False) for fp in in_fps], how="diagonal_relaxed").sort(
                by=["patient_id", "time"], maintain_order=True, multithreaded=False
            )

        logger.info(f"Merging files to {str(out_fp.resolve())}")
        result_computed, _ = rwlock_wrap(
            in_dir,
            out_fp,
            read_fn,
            write_lazyframe,
            identity_fn,
            do_return=False,
            do_overwrite=cfg.do_overwrite,
        )

        if result_computed:
            logger.info(f"Cleaning up subsharded files in {str(subshard_dir.resolve())}/*.")
            for fp in in_fps:
                if fp.exists():
                    fp.unlink()
            subshard_dir.rmdir()

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":
    main()
