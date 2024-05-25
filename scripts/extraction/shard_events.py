#!/usr/bin/env python

import gzip
import random
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import (
    get_shard_prefix,
    hydra_loguru_init,
    write_lazyframe,
)

ROW_IDX_NAME = "__row_idx"


def scan_with_row_idx(fp: Path) -> pl.LazyFrame:
    match "".join(fp.suffixes):
        case ".csv.gz":
            logger.debug(f"Reading {fp} as compressed CSV.")
            logger.warning("Reading compressed CSV files may be slow and limit parallelizability.")

            def reader(fp: Path, **kwargs) -> pl.LazyFrame:
                return pl.read_csv(gzip.open(fp, mode="rb"), infer_schema_length=1_000_000, **kwargs).lazy()

        case ".csv":
            logger.debug(f"Reading {fp} as CSV.")
            reader = pl.scan_csv
        case ".parquet":
            logger.debug(f"Reading {fp} as Parquet.")
            reader = pl.scan_parquet
        case _:
            raise ValueError(f"Unsupported file type: {fp.suffix}")

    return reader(fp, row_index_name=ROW_IDX_NAME)


def filter_to_row_chunk(df: pl.LazyFrame, start: int, end: int) -> pl.LazyFrame:
    return df.filter(pl.col(ROW_IDX_NAME).is_between(start, end, closed="left")).drop(ROW_IDX_NAME)


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Runs the input data re-sharding process. Can be parallelized across output shards.

    Output shards are simply row-chunks of the input data. There is no randomization or re-ordering of the
    input data. Read contention on the input files may render additional parallelism beyond one worker per
    input file ineffective.
    """

    hydra_loguru_init()

    raw_cohort_dir = Path(cfg.raw_cohort_dir)
    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)
    row_chunksize = cfg.row_chunksize

    seen_files = set()
    input_files_to_subshard = []
    for fmt in ["parquet", "csv", "csv.gz"]:
        files_in_fmt = list(raw_cohort_dir.glob(f"**/*.{fmt}"))
        for f in files_in_fmt:
            if get_shard_prefix(raw_cohort_dir, f) in seen_files:
                logger.warning(f"Skipping {f} as it has already been added in a preferred format.")
            else:
                input_files_to_subshard.append(f)
                seen_files.add(get_shard_prefix(raw_cohort_dir, f))

    logger.info(f"Starting event sub-sharding. Sub-sharding {len(input_files_to_subshard)} files.")
    logger.info(
        f"Will read raw data from {str(raw_cohort_dir.resolve())}/$IN_FILE.parquet and write sub-sharded "
        f"data to {str(MEDS_cohort_dir.resolve())}/sub_sharded/$IN_FILE/$ROW_START-$ROW_END.parquet"
    )

    random.shuffle(input_files_to_subshard)

    for input_file in input_files_to_subshard:
        out_dir = MEDS_cohort_dir / "sub_sharded" / get_shard_prefix(raw_cohort_dir, input_file)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {input_file} to {out_dir}.")

        df = scan_with_row_idx(input_file)
        row_count = df.select(pl.len()).collect().item()

        row_shards = list(range(0, row_count, row_chunksize))
        random.shuffle(row_shards)
        logger.info(f"Splitting {input_file} into {len(row_shards)} row-chunks of size {row_chunksize}.")

        datetime.now()
        for st in row_shards:
            end = min(st + row_chunksize, row_count)
            out_fp = out_dir / f"[{st}-{end}).parquet"

            compute_fn = partial(filter_to_row_chunk, start=st, end=end)
            logger.info(f"Writing {input_file} row-chunk [{st}-{end}) to {out_fp}.")
            rwlock_wrap(
                input_file,
                out_fp,
                scan_with_row_idx,
                write_lazyframe,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
            )


if __name__ == "__main__":
    main()
