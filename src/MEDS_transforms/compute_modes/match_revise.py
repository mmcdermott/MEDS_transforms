"""This module contains utilities to run stages in "Match-revise" mode.

Match-revise mode allows one to dynamically select a set of filters and different configuration parameters
such that a given stage will run with the corresponding parameters on each of the sets of data in the input
dataframe that matches the given filter.
"""

import logging
from enum import StrEnum, auto
from functools import wraps

import hydra
import polars as pl
from meds import subject_id_field
from omegaconf import DictConfig, ListConfig

from ..dataframe import DF_T
from ..parser import is_matcher, matcher_to_expr
from .compute_fn import ANY_COMPUTE_FN_T, COMPUTE_FN_T, bind_compute_fn

logger = logging.getLogger(__name__)

MATCH_REVISE_KEY = "_match_revise"
MATCHER_KEY = "_matcher"
MATCH_REVISE_MODE_KEY = "_match_revise_mode"


class MatchReviseMode(StrEnum):
    """The different modes for match and revise operations.

    Future modes to be considered, match and add, multi-match and add, filter and revise, multi-filter and
    revise.

    Attributes:
        MATCH_AND_REVISE: The match and revise mode, which iterates through the list of matcher/function pairs
            and filters the input DataFrame for rows that match the matcher and applies a local compute
            function to the filtered DataFrame. The DataFrame to be matched in future iterations is restricted
            to only those rows that have not yet been matched. The unmatched dataframe at the end of the
            operation is concatenated with the outputs of all intermediate dataframes.
        MULTI_MATCH_AND_REVISE: The match and revise mode, which iterates through the list of matcher/function
            pairs and filters the input DataFrame for rows that match the matcher and applies a local compute
            function to the filtered DataFrame. The DataFrame to be matched in future iterations is the entire
            raw dataframe, including rows that have already been matched. The portion of the dataframe that
            didn't match anything on input is concatenated with the outputs of all intermediate dataframes.
    """

    MATCH_AND_REVISE = auto()
    MULTI_MATCH_AND_REVISE = auto()


def is_match_revise(stage_cfg: DictConfig) -> bool:
    """Check if the stage configuration is in a match and revise format.

    Examples:
        >>> is_match_revise(DictConfig({"_match_revise": []}))
        False
        >>> is_match_revise(DictConfig({"_match_revise": [{"_matcher": {"code": "CODE//TEMP"}}]}))
        True
        >>> is_match_revise(DictConfig({"foo": "bar"}))
        False
    """
    return bool(stage_cfg.get(MATCH_REVISE_KEY, False))


def validate_match_revise(stage_cfg: DictConfig):
    """Validate that the stage configuration is in a match and revise format.

    Examples:
        >>> validate_match_revise(DictConfig({"foo": []}))
        Traceback (most recent call last):
            ...
        ValueError: Stage configuration must contain a _match_revise key
        >>> validate_match_revise(DictConfig({"_match_revise": "foo"}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise options must be a list, got <class 'str'>
        >>> validate_match_revise(DictConfig({"_match_revise": [1]}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise config 0 must be a dict, got <class 'int'>
        >>> validate_match_revise(DictConfig({"_match_revise": [{"_matcher": {"foo": "bar"}}, 1]}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise config 1 must be a dict, got <class 'int'>
        >>> validate_match_revise(DictConfig({"_match_revise": [{"foo": "bar"}]}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise config 0 must contain a _matcher key
        >>> validate_match_revise(DictConfig({"_match_revise": [{"_matcher": {32: "bar"}}]}))
        Traceback (most recent call last):
            ...
        ValueError: Match revise config 0 must contain a valid matcher in _matcher: ...
        >>> validate_match_revise(DictConfig({"_match_revise": [{"_matcher": {"code": "CODE//TEMP"}}]}))
    """

    if MATCH_REVISE_KEY not in stage_cfg:
        raise ValueError(f"Stage configuration must contain a {MATCH_REVISE_KEY} key")

    match_revise_options = stage_cfg[MATCH_REVISE_KEY]
    if not isinstance(match_revise_options, list | ListConfig):
        raise ValueError(f"Match revise options must be a list, got {type(match_revise_options)}")

    for i, match_revise_cfg in enumerate(match_revise_options):
        if not isinstance(match_revise_cfg, dict | DictConfig):
            raise ValueError(f"Match revise config {i} must be a dict, got {type(match_revise_cfg)}")

        if MATCHER_KEY not in match_revise_cfg:
            raise ValueError(f"Match revise config {i} must contain a {MATCHER_KEY} key")

        matcher_valid, matcher_errs = is_matcher(match_revise_cfg[MATCHER_KEY])
        if not matcher_valid:
            raise ValueError(
                f"Match revise config {i} must contain a valid matcher in {MATCHER_KEY}: {matcher_errs}"
            )


def match_revise_fntr(cfg: DictConfig, stage_cfg: DictConfig, compute_fn: ANY_COMPUTE_FN_T) -> COMPUTE_FN_T:
    """A functor that creates a match & revise compute function based on the given configuration.

    Stage configurations for match & revise must be in a match and revise format. Consider the below example,
    showing the ``stage_cfg`` object in ``yaml`` format:

        ```yaml
        global_arg_1: "foo"
        _match_revise:
          - _matcher: {code: "CODE//BAR"}
            local_arg_1: "bar"
          - _matcher: {code: "CODE//BAZ"}
            local_arg_1: "baz"
        ```

    This configuration will create a match & revise compute function that will filter the input DataFrame for
    rows that match the ``CODE//BAR`` code and apply the compute function with the ``local_arg_1=bar``
    parameter, and then filter the input DataFrame for rows that match the ``CODE//BAZ`` code and apply the
    compute function with the ``local_arg_1=baz`` parameter. Both of these local compute functions will be
    applied to the input DataFrame in sequence, and the resulting DataFrames will be concatenated alongside
    any of the dataframe that matches no matcher (which will be left unmodified) and merged in a sorted way
    that respects the ``subject_id``, ``time`` ordering first, then the order of the match & revise blocks
    themselves, then the order of the rows in each match & revise block output. Each local compute function
    will also use the ``global_arg_1=foo`` parameter.

    Args:
        cfg: The DictConfig configuration object.
        stage_cfg: The DictConfig stage configuration object. This stage configuration must be in a match and
            revise format, meaning it must have a key ``"_match_revise"`` that contains a list of local match
            & revise configurations. Each local match & revise configuration must contain a key ``"_matcher"``
            which links to the matcher configuration to use to filter the input DataFrame for the local
            compute execution, and all other keys are local configuration parameters to be used in the local
            compute execution.
        compute_fn: The compute function to bind to the match & revise configuration local arguments.

    Returns:
        A function that applies the match & revise compute function to the input DataFrame.

    Raises:
        ValueError: If the stage configuration is not in a match and revise format.

    Examples:
        >>> df = pl.DataFrame(
        ...     {
        ...         "subject_id": [1, 1, 1, 2, 2, 2],
        ...         "time": [1, 2, 2, 1, 1, 2],
        ...         "initial_idx": [0, 1, 2, 3, 4, 5],
        ...         "code": [
        ...             "FINAL",
        ...             "CODE//TEMP_2",
        ...             "CODE//TEMP_1",
        ...             "FINAL",
        ...             "CODE//TEMP_2",
        ...             "CODE//TEMP_1",
        ...         ],
        ...     }
        ... )
        >>> def compute_fn(df: pl.DataFrame, stage_cfg: DictConfig) -> pl.DataFrame:
        ...     return df.with_columns(
        ...         pl.col("code").str.slice(0, len("CODE//"))
        ...         + stage_cfg.local_code_mid
        ...         + "//"
        ...         + stage_cfg.global_code_end
        ...     )
        >>> stage_cfg = DictConfig(
        ...     {
        ...         "global_code_end": "foo",
        ...         "_match_revise": [
        ...             {"_matcher": {"code": "CODE//TEMP_1"}, "local_code_mid": "bar"},
        ...             {"_matcher": {"code": "CODE//TEMP_2"}, "local_code_mid": "baz"},
        ...         ],
        ...     }
        ... )
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        >>> match_revise_fn(df.lazy()).collect()
        shape: (6, 4)
        ┌────────────┬──────┬─────────────┬────────────────┐
        │ subject_id ┆ time ┆ initial_idx ┆ code           │
        │ ---        ┆ ---  ┆ ---         ┆ ---            │
        │ i64        ┆ i64  ┆ i64         ┆ str            │
        ╞════════════╪══════╪═════════════╪════════════════╡
        │ 1          ┆ 1    ┆ 0           ┆ FINAL          │
        │ 1          ┆ 2    ┆ 2           ┆ CODE//bar//foo │
        │ 1          ┆ 2    ┆ 1           ┆ CODE//baz//foo │
        │ 2          ┆ 1    ┆ 4           ┆ CODE//baz//foo │
        │ 2          ┆ 1    ┆ 3           ┆ FINAL          │
        │ 2          ┆ 2    ┆ 5           ┆ CODE//bar//foo │
        └────────────┴──────┴─────────────┴────────────────┘
        >>> stage_cfg = DictConfig(
        ...     {
        ...         "global_code_end": "foo",
        ...         "_match_revise": [
        ...             {"_matcher": {"code": "CODE//TEMP_2"}, "local_code_mid": "bizz"},
        ...             {
        ...                 "_matcher": {"code": "CODE//TEMP_1"},
        ...                 "local_code_mid": "foo",
        ...                 "global_code_end": "bar",
        ...             },
        ...         ],
        ...     }
        ... )
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        >>> match_revise_fn(df.lazy()).collect()
        shape: (6, 4)
        ┌────────────┬──────┬─────────────┬─────────────────┐
        │ subject_id ┆ time ┆ initial_idx ┆ code            │
        │ ---        ┆ ---  ┆ ---         ┆ ---             │
        │ i64        ┆ i64  ┆ i64         ┆ str             │
        ╞════════════╪══════╪═════════════╪═════════════════╡
        │ 1          ┆ 1    ┆ 0           ┆ FINAL           │
        │ 1          ┆ 2    ┆ 1           ┆ CODE//bizz//foo │
        │ 1          ┆ 2    ┆ 2           ┆ CODE//foo//bar  │
        │ 2          ┆ 1    ┆ 4           ┆ CODE//bizz//foo │
        │ 2          ┆ 1    ┆ 3           ┆ FINAL           │
        │ 2          ┆ 2    ┆ 5           ┆ CODE//foo//bar  │
        └────────────┴──────┴─────────────┴─────────────────┘
        >>> stage_cfg = DictConfig(
        ...     {"global_code_end": "foo", "_match_revise": [{"_matcher": {"missing": "CODE//TEMP_2"}}]}
        ... )
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        >>> match_revise_fn(df.lazy()).collect()
        Traceback (most recent call last):
            ...
        ValueError: Missing needed columns {'missing'} for local matcher 0:
            [(col("missing")) == ("CODE//TEMP_2")].all_horizontal()
        Columns available: 'code', 'initial_idx', 'subject_id', 'time'

        It will throw an error if the match and revise configuration is missing.
        >>> stage_cfg = DictConfig({"global_code_end": "foo"})
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        Traceback (most recent call last):
            ...
        ValueError: Invalid match and revise configuration...

        It does not accept invalid modes.
        >>> stage_cfg = DictConfig(
        ...     {
        ...         "global_code_end": "foo",
        ...         "_match_revise_mode": "foobar",
        ...         "_match_revise": [{"_matcher": {"code": "CODE//TEMP_2"}}],
        ...     }
        ... )
        >>> cfg = DictConfig({"stage_cfg": stage_cfg})
        >>> match_revise_fn = match_revise_fntr(cfg, stage_cfg, compute_fn)
        Traceback (most recent call last):
            ...
        ValueError: Invalid match and revise mode: foobar
    """
    try:
        validate_match_revise(stage_cfg)
    except ValueError as e:
        raise ValueError("Invalid match and revise configuration") from e

    stage_cfg = hydra.utils.instantiate(stage_cfg)

    match_revise_mode = stage_cfg.pop(MATCH_REVISE_MODE_KEY, "match_and_revise")
    if match_revise_mode not in {x.value for x in MatchReviseMode}:
        raise ValueError(f"Invalid match and revise mode: {match_revise_mode}")

    matchers_and_fns = []
    for match_revise_cfg in stage_cfg.pop(MATCH_REVISE_KEY):
        matcher, cols = matcher_to_expr(match_revise_cfg.pop(MATCHER_KEY))
        local_stage_cfg = DictConfig({**stage_cfg, **match_revise_cfg})
        local_compute_fn = bind_compute_fn(cfg, local_stage_cfg, compute_fn)

        matchers_and_fns.append((matcher, cols, local_compute_fn))

    @wraps(compute_fn)
    def match_revise_fn(df: DF_T) -> DF_T:
        matchable_df = df
        cols = set(df.collect_schema().names())

        revision_parts = []
        final_part_filters = []
        for i, (matcher_expr, need_cols, local_compute_fn) in enumerate(matchers_and_fns):
            if not need_cols.issubset(cols):
                cols_str = "', '".join(x for x in sorted(cols))
                raise ValueError(
                    f"Missing needed columns {need_cols - cols} for local matcher {i}: "
                    f"{matcher_expr}\nColumns available: '{cols_str}'"
                )
            matched_df = matchable_df.filter(matcher_expr)

            match match_revise_mode:
                case MatchReviseMode.MATCH_AND_REVISE:
                    matchable_df = matchable_df.filter(~matcher_expr)
                case MatchReviseMode.MULTI_MATCH_AND_REVISE:
                    final_part_filters.append(~matcher_expr)

            revision_parts.append(local_compute_fn(matched_df))

        if final_part_filters:
            revision_parts.append(matchable_df.filter(pl.all_horizontal(final_part_filters)))
        else:
            revision_parts.append(matchable_df)
        return pl.concat(revision_parts, how="vertical").sort([subject_id_field, "time"], maintain_order=True)

    return match_revise_fn
