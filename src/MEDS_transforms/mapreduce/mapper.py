"""Basic utilities for parallelizable map operations on sharded MEDS datasets with caching and locking."""

from collections.abc import Callable, Generator
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
from loguru import logger
from omegaconf import DictConfig

from ..extract.parser import is_matcher, matcher_to_expr
from ..utils import stage_init, write_lazyframe
from .utils import rwlock_wrap, shard_iterator

DF_T = TypeVar("DF_T")

MAP_FN_T = Callable[[DF_T], DF_T]
SHARD_GEN_T = Generator[tuple[Path, Path], None, None]
SHARD_ITR_FNTR_T = Callable[[DictConfig], SHARD_GEN_T]


def identity_fn(df: Any) -> Any:
    return df


def read_and_filter_fntr(filter_expr: pl.Expr, read_fn: Callable[[Path], DF_T]) -> Callable[[Path], DF_T]:
    def read_and_filter(in_fp: Path) -> DF_T:
        return read_fn(in_fp).filter(filter_expr)

    return read_and_filter


MATCH_REVISE_KEY = "_match_revise"
MATCHER_KEY = "_matcher"


def is_match_revise(stage_cfg: DictConfig) -> bool:
    return stage_cfg.get(MATCH_REVISE_KEY, False)


def validate_match_revise(stage_cfg: DictConfig):
    match_revise_options = stage_cfg[MATCH_REVISE_KEY]
    if not isinstance(match_revise_options, (list, ListConfig)):
        raise ValueError(f"Match revise options must be a list, got {type(match_revise_options)}")

    for match_revise_cfg in match_revise_options:
        if not isinstance(match_revise_cfg, (dict, DictConfig)):
            raise ValueError(f"Match revise config must be a dict, got {type(match_revise_cfg)}")

        if MATCHER_KEY not in match_revise_cfg:
            raise ValueError(f"Match revise config must contain a {MATCHER_KEY} key")

        if not is_matcher(match_revise_cfg[MATCHER_KEY]):
            raise ValueError(f"Match revise config must contain a valid matcher in {MATCHER_KEY}")


def match_revise_fntr(matcher_expr: pl.Expr, compute_fn: Callable[[DF_T], DF_T]) -> Callable[[DF_T], DF_T]:
    @wraps(compute_fn)
    def match_revise_fn(df: DF_T) -> DF_T:
        cols = df.collect_schema().names
        idx_col = "_row_idx"
        while idx_col in cols:
            idx_col = f"_{idx_col}"

        df = df.with_row_index(idx_col)

        matches = df.filter(matcher_expr)
        revised = compute_fn(matches)
        return compute_fn(df.filter(matcher_expr))

    return match_revise_fn


def get_match_revise_compute_fn(stage_cfg: DictConfig, compute_fn: MAP_FN_T) -> MAP_FN_T:
    if not is_match_revise(stage_cfg):
        return compute_fn

    validate_match_revise(stage_cfg)

    match_revise_options = stage_cfg[MATCH_REVISE_KEY]
    out_compute_fn = []
    for match_revise_cfg in match_revise_options:
        matcher = matcher_to_expr(match_revise_cfg[MATCHER_KEY])
        compute_fn = (partial(pl.filter, matcher),) + compute_fn


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
        read_fn = read_and_filter_fntr(pl.col("patient_id").isin(split_patients), read_fn)
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
            compute_fn,
            do_return=False,
            do_overwrite=cfg.do_overwrite,
        )
        all_out_fps.append(out_fp)

    logger.info(f"Finished mapping in {datetime.now() - start}")
    return all_out_fps
