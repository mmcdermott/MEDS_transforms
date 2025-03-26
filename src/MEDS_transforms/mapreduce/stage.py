"""Basic code for a mapreduce stage."""

import logging
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import polars as pl
from meds import subject_id_field, subject_splits_filepath
from omegaconf import DictConfig

from ..utils import stage_init, write_lazyframe
from .compute_fn import ANY_COMPUTE_FN_T, COMPUTE_FN_T, bind_compute_fn
from .mapper import map_over
from .match_revise import is_match_revise, match_revise_fntr
from .read_fn import read_and_filter_fntr
from .shard_iteration import SHARD_ITR_FNTR_T, shard_iterator
from .types import DF_T

logger = logging.getLogger(__name__)


def resolve_compute_fn(cfg: DictConfig, compute_fn: ANY_COMPUTE_FN_T | None) -> COMPUTE_FN_T:
    if is_match_revise(cfg.stage_cfg):
        return match_revise_fntr(cfg, cfg.stage_cfg, compute_fn)
    else:
        return bind_compute_fn(cfg, cfg.stage_cfg, compute_fn)


def map_stage(
    cfg: DictConfig,
    compute_fn: COMPUTE_FN_T | None = None,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
) -> list[Path]:
    stage_init(cfg)

    start = datetime.now()

    train_only = cfg.stage_cfg.get("train_only", False)

    shards, includes_only_train = shard_iterator_fntr(cfg)

    if train_only:
        split_fp = Path(cfg.input_dir) / subject_splits_filepath
        if includes_only_train:
            logger.info(
                f"Processing train split only via shard prefix. Not filtering with {str(split_fp.resolve())}."
            )
        elif split_fp.exists():
            logger.info(f"Processing train split only by filtering read dfs via {str(split_fp.resolve())}")
            train_subjects = (
                pl.scan_parquet(split_fp)
                .filter(pl.col("split") == "train")
                .select(subject_id_field)
                .collect()[subject_id_field]
                .to_list()
            )
            read_fn = read_and_filter_fntr(pl.col("subject_id").is_in(train_subjects), read_fn)
        else:
            raise FileNotFoundError(
                f"Train split requested, but shard prefixes can't be used and "
                f"subject split file not found at {str(split_fp.resolve())}."
            )
    elif includes_only_train:  # pragma: no cover
        raise ValueError("All splits should be used, but shard iterator is returning only train splits?!?")

    compute_fn = resolve_compute_fn(cfg, compute_fn)

    all_out_fps = list(
        map_over(
            shards=shards,
            read_fn=read_fn,
            write_fn=write_fn,
            compute_fn=compute_fn,
            do_overwrite=cfg.do_overwrite,
        )
    )
    logger.info(f"Finished mapping in {datetime.now() - start}")
    return all_out_fps
