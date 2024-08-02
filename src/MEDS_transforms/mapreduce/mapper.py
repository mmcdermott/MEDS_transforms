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


def read_and_filter_fntr(patients: list[int], read_fn: Callable[[Path], DF_T]) -> Callable[[Path], DF_T]:
    def read_and_filter(in_fp: Path) -> DF_T:
        df = read_fn(in_fp)
        return df.filter(pl.col("patient_id").isin(patients))

    return read_and_filter


def map_over(
    cfg: DictConfig,
    compute_fn: MAP_FN_T | None = None,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
) -> list[Path]:
    stage_init(cfg)

    start = datetime.now()

    if compute_fn is None:
        compute_fn = identity_fn

    if not isinstance(compute_fn, tuple):
        compute_fn = (compute_fn,)

    process_split = cfg.stage_cfg.get("process_split", None)
    split_fp = Path(cfg.input_dir) / "metadata" / "patient_split.parquet"
    shards_map_fp = Path(cfg.shards_map_fp) if "shards_map_fp" in cfg else None
    if process_split and split_fp.exists():
        split_patients = (
            pl.scan_parquet(split_fp)
            .filter(pl.col("split") == process_split)
            .select(pl.col("patient_id"))
            .collect()
            .to_list()
        )
        read_fn = read_and_filter_fntr(split_patients, read_fn)
    elif process_split and shards_map_fp and shards_map_fp.exists():
        logger.warning(
            f"Split {process_split} requested, but no patient split file found at {str(split_fp)}. "
            f"Assuming this is handled through shard filtering."
        )
    elif process_split:
        raise ValueError(
            f"Split {process_split} requested, but no patient split file found at {str(split_fp)}."
        )

    all_out_fps = []
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
        all_out_fps.append(out_fp)

    logger.info(f"Finished mapping in {datetime.now() - start}")
    return all_out_fps
