"""Basic code for a mapreduce stage."""

import logging
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import polars as pl
from meds import code_field, subject_id_field, subject_splits_filepath
from omegaconf import DictConfig

from ..utils import stage_init, write_lazyframe
from .compute_fn import ANY_COMPUTE_FN_T, COMPUTE_FN_T, bind_compute_fn
from .mapper import map_over
from .match_revise import is_match_revise, match_revise_fntr
from .read_fn import read_and_filter_fntr
from .reducer import REDUCE_FN_T, reduce_over
from .shard_iteration import SHARD_ITR_FNTR_T, shard_iterator
from .types import DF_T

logger = logging.getLogger(__name__)


def resolve_mapper_fn(cfg: DictConfig, map_fn: ANY_COMPUTE_FN_T | None) -> COMPUTE_FN_T:
    if is_match_revise(cfg.stage_cfg):
        return match_revise_fntr(cfg, cfg.stage_cfg, map_fn)
    else:
        return bind_compute_fn(cfg, cfg.stage_cfg, map_fn)


def map_stage(
    cfg: DictConfig,
    map_fn: COMPUTE_FN_T | None = None,
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

    map_fn = resolve_mapper_fn(cfg, map_fn)

    all_out_fps = map_over(
        shards=shards,
        read_fn=read_fn,
        write_fn=write_fn,
        map_fn=map_fn,
        do_overwrite=cfg.do_overwrite,
    )
    logger.info(f"Finished mapping in {datetime.now() - start}")
    return all_out_fps


def join_and_replace(new: pl.DataFrame, old: pl.DataFrame, join_cols: list[str]) -> pl.DataFrame:
    """Join two dataframes and replace the old columns with the new columns."""
    return new.join(
        old.drop(*[c for c in old.columns if c in set(new.columns) - set(join_cols)]),
        on=join_cols,
        how="left",
        coalesce=True,
    )


def mapreduce_stage(
    cfg: DictConfig,
    map_fn: ANY_COMPUTE_FN_T,
    reduce_fn: REDUCE_FN_T,
    merge_fn: REDUCE_FN_T | None = None,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
) -> Path:

    map_stage_out_fps = map_stage(
        cfg=cfg, map_fn=map_fn, read_fn=read_fn, write_fn=write_fn, shard_iterator_fntr=shard_iterator
    )

    if cfg.worker != 0:
        logger.info(f"Mapping completed. Exiting as am worker {cfg.worker}, not reducer (0).")
        return

    logger.info("Starting reduction process")
    start = datetime.now()

    merge_fp = Path(cfg.stage_cfg.metadata_input_dir) / "codes.parquet"
    reduce_stage_out_fp = Path(cfg.stage_cfg.reducer_output_dir) / "codes.parquet"

    if merge_fn is None:
        join_cols = [code_field, *cfg.get("code_modifier_cols", [])]
        merge_fn = partial(join_and_replace, join_cols=join_cols)

    reduce_fn = bind_compute_fn(cfg, cfg.stage_cfg, reduce_fn)

    reduce_over(
        in_fps=map_stage_out_fps,
        out_fp=reduce_stage_out_fp,
        polling_time=cfg.polling_time,
        read_fn=read_fn,
        write_fn=write_fn,
        reduce_fn=reduce_fn,
        merge_fp=merge_fp,
        merge_fn=merge_fn,
        do_overwrite=cfg.do_overwrite,
    )
    logger.info(f"Finished reduction in {datetime.now() - start}")

    return reduce_stage_out_fp
