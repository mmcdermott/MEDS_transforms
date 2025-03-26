"""Basic utilities for serialized reduce operations on sharded MEDS datasets with caching and locking."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

import polars.selectors as cs

from .read_fn import READ_FN_T
from .rwlock import WRITE_FN_T
from .types import DF_T

logger = logging.getLogger(__name__)

REDUCE_FN_T = Callable[[DF_T, DF_T | None], DF_T]


def reduce_over(
    in_fps: list[Path],
    out_fp: Path,
    polling_time: float,
    read_fn: READ_FN_T,
    write_fn: WRITE_FN_T,
    reduce_fn: REDUCE_FN_T,
    merge_fp: Path,
    merge_fn: REDUCE_FN_T,
    do_overwrite: bool = False,
) -> Path:
    if out_fp.is_file() and not do_overwrite:
        raise FileExistsError(f"Output file already exists: {str(out_fp.resolve())}")

    while not all(fp.is_file() for fp in in_fps):
        logger.info("Waiting to begin reduction for all files to be written...")
        time.sleep(polling_time)

    reduced = reduce_fn(*[read_fn(fp) for fp in in_fps]).with_columns(cs.numeric().shrink_dtype().name.keep())

    if merge_fp.is_file():
        reduced = merge_fn(reduced, read_fn(merge_fp))

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    write_fn(reduced, out_fp)
    return out_fp
