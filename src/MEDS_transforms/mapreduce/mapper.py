"""Basic utilities for parallelizable map operations on sharded MEDS datasets with caching and locking."""

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import polars as pl

from ..utils import write_lazyframe
from .compute_fn import COMPUTE_FN_T
from .rwlock import rwlock_wrap
from .shard_iteration import InOutFilePair
from .types import DF_T

logger = logging.getLogger(__name__)


def map_over(
    shards: list[InOutFilePair],
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    compute_fn: COMPUTE_FN_T | None = None,
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    do_overwrite: bool = False,
) -> list[Path]:
    all_out_fps = []
    for in_fp, out_fp in shards:
        logger.info(f"Processing {str(in_fp.resolve())} into {str(out_fp.resolve())}")
        rwlock_wrap(
            in_fp,
            out_fp,
            read_fn,
            write_fn,
            compute_fn,
            do_overwrite=do_overwrite,
        )
        all_out_fps.append(out_fp)
    return all_out_fps
