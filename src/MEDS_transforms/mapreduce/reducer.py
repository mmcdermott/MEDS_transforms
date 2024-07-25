"""Basic utilities for serialized reduce operations on sharded MEDS datasets with caching and locking."""

from collections.abc import Callable, Generator
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TypeVar

import polars as pl
from loguru import logger
from omegaconf import DictConfig

from ..utils import write_lazyframe

DF_T = TypeVar("DF_T")

REDUCE_FN_T = Callable[[DF_T, DF_T | None], DF_T]
SHARD_GEN_T = Generator[tuple[Path, Path], None, None]
SHARD_ITR_FNTR_T = Callable[[DictConfig], SHARD_GEN_T]


def join_merger_fntr(cfg: DictConfig) -> REDUCE_FN_T:
    raise NotImplementedError("Join merging not yet implemented")


def take_new_and_error(new: DF_T, disk: DF_T | None) -> DF_T:
    if disk is not None:
        raise FileExistsError(f"File already exists on disk")
    return new


def reduce_over(
    cfg: DictConfig,
    all_out_fps: list[Path],
    compute_fn: REDUCE_FN_T,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    merge_fn: REDUCE_FN_T | None = None,
) -> list[Path]:
    if cfg.worker != 0:
        logger.info("Mapping stage completed. Exiting")
        return

    logger.info("Starting reduction process")

    start = datetime.now()

    while not all(fp.is_file() for fp in all_out_fps):
        logger.info("Waiting to begin reduction for all files to be written...")
        time.sleep(cfg.polling_time)

    reduced = reducer_fn(*[read_fn(fp) for fp in all_out_fps])

    reducer_fp = Path(cfg.stage_cfg.reduced_output_fp)
    on_disk = read_fn(reducer_fp) if reducer_fp.is_file() else None

    if merge_fn is None:
        merge_fn = join_merger_fntr(cfg)

    reduced = merge_fn(reduced, on_disk)
    write_fn(reduced, reducer_fp)

    logger.info(f"Finished reduction in {datetime.now() - start}")
    return reduced
