"""Basic utilities for parallelizable map operations on sharded MEDS datasets with caching and locking."""

import copy
import inspect
from collections.abc import Callable, Generator
from datetime import datetime
from enum import Enum, auto
from functools import partial, wraps
from pathlib import Path
from typing import Any, NotRequired, TypedDict, TypeVar

import polars as pl
from loguru import logger
from omegaconf import DictConfig, ListConfig

from ..extract.parser import is_matcher, matcher_to_expr
from ..utils import stage_init, write_lazyframe
from .utils import rwlock_wrap, shard_iterator

DF_T = TypeVar("DF_T")

COMPUTE_FN_T = Callable[[DF_T], DF_T]
COMPUTE_FN_UNBOUND_T = Callable[..., DF_T]
COMPUTE_FNTR_T = Callable[..., COMPUTE_FN_T]
ANY_COMPUTE_FN_T = COMPUTE_FN_T | COMPUTE_FN_UNBOUND_T | COMPUTE_FNTR_T


class ComputeFnArgs(TypedDict, total=False):
    df: NotRequired[DF_T]
    cfg: NotRequired[DictConfig]
    stage_cfg: NotRequired[DictConfig]
    code_modifiers: NotRequired[list[str]]
    code_metadata: NotRequired[pl.DataFrame]


SHARD_GEN_T = Generator[tuple[Path, Path], None, None]
SHARD_ITR_FNTR_T = Callable[[DictConfig], SHARD_GEN_T]


class ComputeFnType(Enum):
    """Stores the three types of compute functions that can be used in this utility.

    Attributes:
        DIRECT: A direct compute function that maps a DataFrame to a DataFrame.
        UNBOUND: An unbound compute function that needs to be called with additional parameters in addition to
            the input DataFrame, but still returns a DataFrame.
        FUNCTOR: A functor that needs to be called with non "df" args to return a direct compute function.
    """

    DIRECT = auto()
    UNBOUND = auto()
    FUNCTOR = auto()


def compute_fn_type(compute_fn: ANY_COMPUTE_FN_T) -> ComputeFnType | None:
    """Determine the type of a compute function, or return None if the type matches no compute function type.

    Returns the type of the compute function:
      - ComputeFnType.DIRECT if the compute function is a direct compute function -- i.e., if it takes only a
        single parameter named ``"df"`` and if the return annotation (if one exists) is not a
        `collections.abc.Callable` type annotation.
      - ComputeFnType.UNBOUND if the compute function is an unbound compute function -- i.e., if it takes a
        ``"df"`` parameter and at least one other parameter among the set of allowed additional or functor
        parameters, and if the return annotation (if one exists) is not a `collections.abc.Callable` type
        annotation.
      - ComputeFnType.FUNCTOR if the compute function is a functor that returns a direct compute function when
        called with the appropriate parameters -- i.e., if it takes no ``"df"`` parameter, if all parameters
        are among the allowed functor parameters, and if the return annotation (if one exists) is a
        `collections.abc.Callable` type annotation.

    Allowed functor parameters are:
      - "cfg" for the DictConfig configuration object.
      - "stage_cfg" for the DictConfig stage configuration object.
      - "code_modifiers" for a list of code modifier columns.
      - "code_metadata" for a Polars DataFrame of code metadata

    Args:
        compute_fn: The compute function to check.

    Examples:
        >>> def compute_fn(df: pl.DataFrame) -> pl.DataFrame: return df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.DIRECT: 1>
        >>> def compute_fn(df: pl.DataFrame) -> dict[Any, list[Any]]: return df.to_dict(as_series=False)
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.DIRECT: 1>
        >>> def compute_fn(df: pl.DataFrame): return None
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.DIRECT: 1>
        >>> def compute_fn(foo: pl.DataFrame): return None
        >>> compute_fn_type(compute_fn) is None
        True
        >>> def compute_fn(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame: return df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.UNBOUND: 2>
        >>> def compute_fn(df: pl.DataFrame, foo: DictConfig) -> pl.DataFrame: return df
        >>> compute_fn_type(compute_fn) is None
        True
        >>> def compute_fn(df: pl.LazyFrame, cfg: DictConfig): return df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.UNBOUND: 2>
        >>> def compute_fn(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]: return lambda df: df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.FUNCTOR: 3>
        >>> def compute_fn(cfg: DictConfig): return lambda df: df
        >>> compute_fn_type(compute_fn)
        <ComputeFnType.FUNCTOR: 3>
        >>> def compute_fn(df: pl.DataFrame, cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
        ...     return lambda df: df
        >>> compute_fn_type(compute_fn) is None
        True
    """
    sig = inspect.signature(compute_fn)

    allowed_params = {*ComputeFnArgs.__required_keys__, *ComputeFnArgs.__optional_keys__}
    all_params_allowed = all(param in allowed_params for param in sig.parameters.keys())
    if not all_params_allowed:
        return None

    has_df_param = "df" in sig.parameters
    has_only_df_param = has_df_param and (len(sig.parameters) == 1)
    has_return_annotation = sig.return_annotation.__name__ != "_empty"
    has_callable_return = sig.return_annotation.__name__ == "Callable"

    if has_only_df_param:
        if has_return_annotation:
            return ComputeFnType.DIRECT if not has_callable_return else None
        return ComputeFnType.DIRECT
    elif has_df_param:
        if has_return_annotation:
            return ComputeFnType.UNBOUND if not has_callable_return else None
        return ComputeFnType.UNBOUND
    else:
        if has_return_annotation:
            return ComputeFnType.FUNCTOR if has_callable_return else None
        return ComputeFnType.FUNCTOR


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


def match_revise_fntr(cfg: DictConfig, stage_cfg: DictConfig, compute_fn: ANY_COMPUTE_FN_T) -> COMPUTE_FN_T:
    """TODO.

    Args:

    Returns:

    Examples:
        >>> raise NotImplementedError("TODO: Add examples")
    """
    stage_cfg = copy.deepcopy(stage_cfg)

    if not is_match_revise(stage_cfg):
        return bind_compute_fn(cfg, stage_cfg, compute_fn)

    validate_match_revise(stage_cfg)

    matchers_and_fns = []
    for match_revise_cfg in stage_cfg.pop(MATCH_REVISE_KEY):
        matcher = matcher_to_expr(match_revise_cfg.pop(MATCHER_KEY))
        local_stage_cfg = DictConfig({**stage_cfg, **match_revise_cfg})
        local_compute_fn = bind_compute_fn(cfg, local_stage_cfg, compute_fn)

        matchers_and_fns.append((matcher, local_compute_fn))

    @wraps(compute_fn)
    def match_revise_fn(df: DF_T) -> DF_T:
        cols = df.collect_schema().names
        idx_col = "_row_idx"
        while idx_col in cols:
            idx_col = f"_{idx_col}"

        df = df.with_row_index(idx_col)

        unmatched_df = df

        revision_parts = []
        for matcher_expr, local_compute_fn in matchers_and_fns:
            matched_df = unmatched_df.filter(matcher_expr).with_columns(
                pl.col(idx_col).fill_null("forward").name.keep()
            )
            unmatched_df = unmatched_df.filter(~matcher_expr)

            revision_parts = local_compute_fn(matched_df)

        revision_parts.append(unmatched_df)
        return (
            pl.concat(revision_parts, how="vertical")
            .sort(["patient_id", "time", idx_col], maintain_order=True)
            .drop(idx_col)
        )

    return match_revise_fn


def bind_compute_fn(cfg: DictConfig, stage_cfg: DictConfig, compute_fn: ANY_COMPUTE_FN_T) -> COMPUTE_FN_T:
    """Bind the compute function to the appropriate parameters based on the type of the compute function.

    Args:
        cfg: The DictConfig configuration object.
        stage_cfg: The DictConfig stage configuration object. This is separated from the ``cfg`` argument
            because in some cases, such as under the match and revise paradigm, the stage config may be
            modified dynamically under different matcher conditions to yield different compute functions.
        compute_fn: The compute function to bind.

    Returns:
        The compute function bound to the appropriate parameters.

    Raises:
        ValueError: If the compute function is not a valid compute function.

    Examples:
        >>> raise NotImplementedError("TODO: Add examples")
    """

    def fntr_params(compute_fn: ANY_COMPUTE_FN_T) -> ComputeFnArgs:
        compute_fn_params = inspect.signature(compute_fn).parameters
        kwargs = ComputeFnArgs()

        if "cfg" in compute_fn_params:
            kwargs["cfg"] = cfg
        if "stage_cfg" in compute_fn_params:
            kwargs["stage_cfg"] = stage_cfg
        if "code_modifiers" in compute_fn_params:
            code_modifiers = cfg.get("code_modifiers", None)
            kwargs["code_modifiers"] = code_modifiers
        if "code_metadata" in compute_fn_params:
            kwargs["code_metadata"] = pl.read_parquet(
                Path(stage_cfg.metadata_input_dir) / "codes.parquet", use_pyarrow=True
            )
        return kwargs

    if compute_fn is None:
        return identity_fn
    match compute_fn_type(compute_fn):
        case ComputeFnType.DIRECT:
            pass
        case ComputeFnType.UNBOUND:
            compute_fn = partial(compute_fn, **fntr_params(compute_fn))
        case ComputeFnType.FUNCTOR:
            compute_fn = compute_fn(**fntr_params(compute_fn))
        case _:
            raise ValueError("Invalid compute function")

    return compute_fn


def map_over(
    cfg: DictConfig,
    compute_fn: COMPUTE_FN_T | None = None,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
) -> list[Path]:
    stage_init(cfg)

    start = datetime.now()

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

    compute_fn = match_revise_fntr(cfg, cfg.stage_cfg, compute_fn)

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
