"""Basic utilities for parallelizable map operations on sharded MEDS datasets with caching and locking."""

from collections.abc import Callable, Generator
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
from loguru import logger
from omegaconf import DictConfig

from ..utils import stage_init, write_lazyframe
from .utils import rwlock_wrap, shard_iterator

DF_T = TypeVar("DF_T")

MAP_FN_T = Callable[[DF_T], DF_T] | tuple[Callable[[DF_T], DF_T]]
SHARD_GEN_T = Generator[tuple[Path, Path], None, None]
SHARD_ITR_FNTR_T = Callable[[DictConfig], SHARD_GEN_T]


def identity_fn(df: Any) -> Any:
    return df


def map_over(
    cfg: DictConfig,
    compute_fn: MAP_FN_T | None = None,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
):
    stage_init(cfg)

    start = datetime.now()

    if compute_fn is None:
        compute_fn = identity_fn

    if not isinstance(compute_fn, tuple):
        compute_fn = (compute_fn,)

    for in_fp, out_fp in shard_iterator_fntr(cfg):
        logger.info(f"Processing {str(in_fp.resolve())} into {str(out_fp.resolve())}")
        rwlock_wrap(
            in_fp,
            out_fp,
            read_fn,
            write_fn,
            *compute_fn,
            do_return=False,
            cache_intermediate=False,
            do_overwrite=cfg.do_overwrite,
        )

    logger.info(f"Finished mapping in {datetime.now() - start}")
