"""Code for manipulating compute functions w.r.t.

configuration objects or stage configuration objects.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import polars as pl
from omegaconf import DictConfig

from .types import DF_T

COMPUTE_FN_T = Callable[[DF_T], DF_T]
COMPUTE_FN_UNBOUND_T = Callable[..., DF_T]
COMPUTE_FNTR_T = Callable[..., COMPUTE_FN_T]
ANY_COMPUTE_FN_T = COMPUTE_FN_T | COMPUTE_FN_UNBOUND_T | COMPUTE_FNTR_T


class ComputeFnArgs(TypedDict, total=False):
    df: NotRequired[DF_T]
    dfs: NotRequired[DF_T]
    cfg: NotRequired[DictConfig]
    stage_cfg: NotRequired[DictConfig]
    code_modifiers: NotRequired[list[str]]
    code_metadata: NotRequired[pl.DataFrame]


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

    @classmethod
    def from_fn(cls, compute_fn: ANY_COMPUTE_FN_T) -> ComputeFnType | None:
        """Returns the type of a compute function or None if invalid.

        Behavior:
          - Returns `ComputeFnType.DIRECT` if the compute function is a direct compute function -- i.e., if it
            takes only a single parameter named `df` or `dfs` and if the return annotation (if one exists) is
            not a `Callable` type annotation.
          - Returns `ComputeFnType.UNBOUND` if the compute function is an unbound compute function -- i.e., if
            it takes a `df` or `dfs` parameter and at least one other parameter among the set of allowed
            additional or functor parameters, and if the return annotation (if one exists) is not a `Callable`
            type annotation.
          - Returns `ComputeFnType.FUNCTOR` if the compute function is a functor that returns a direct compute
            function when called with the appropriate parameters -- i.e., if it takes no `df` parameter, if
            all parameters are among the allowed functor parameters, and if the return annotation (if one
            exists) is a `Callable` type annotation.

        Allowed functor parameters are:
          - `cfg` for the DictConfig configuration object.
          - `stage_cfg` for the DictConfig stage configuration object.
          - `code_modifiers` for a list of code modifier columns.
          - `code_metadata` for a Polars DataFrame of code metadata

        Args:
            compute_fn: The compute function to check.

        Returns:
            The type of the compute function or `None` if invalid.

        Examples:
            >>> def compute_fn(df: pl.DataFrame) -> pl.DataFrame: return df
            >>> ComputeFnType.from_fn(compute_fn)
            <ComputeFnType.DIRECT: 1>
            >>> def compute_fn(*dfs: pl.DataFrame) -> pl.DataFrame: return pl.concat(dfs)
            >>> ComputeFnType.from_fn(compute_fn)
            <ComputeFnType.DIRECT: 1>
            >>> def compute_fn(df: pl.DataFrame) -> dict[Any, list[Any]]: return df.to_dict(as_series=False)
            >>> ComputeFnType.from_fn(compute_fn)
            <ComputeFnType.DIRECT: 1>
            >>> def compute_fn(df: pl.DataFrame): return None
            >>> ComputeFnType.from_fn(compute_fn)
            <ComputeFnType.DIRECT: 1>
            >>> def compute_fn(foo: pl.DataFrame): return None
            >>> print(ComputeFnType.from_fn(compute_fn))
            None
            >>> def compute_fn(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame: return df
            >>> ComputeFnType.from_fn(compute_fn)
            <ComputeFnType.UNBOUND: 2>
            >>> def compute_fn(df: pl.DataFrame, foo: DictConfig) -> pl.DataFrame: return df
            >>> print(ComputeFnType.from_fn(compute_fn))
            None
            >>> def compute_fn(df: pl.LazyFrame, cfg: DictConfig): return df
            >>> ComputeFnType.from_fn(compute_fn)
            <ComputeFnType.UNBOUND: 2>
            >>> def compute_fn(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
            ...     return lambda df: df
            >>> ComputeFnType.from_fn(compute_fn)
            <ComputeFnType.FUNCTOR: 3>
            >>> def compute_fn(cfg: DictConfig): return lambda df: df
            >>> ComputeFnType.from_fn(compute_fn)
            <ComputeFnType.FUNCTOR: 3>
            >>> def compute_fn(df: pl.DataFrame, cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
            ...     return lambda df: df
            >>> print(ComputeFnType.from_fn(compute_fn))
            None
        """
        sig = inspect.signature(compute_fn)

        allowed_params = {*ComputeFnArgs.__required_keys__, *ComputeFnArgs.__optional_keys__}
        all_params_allowed = all(param in allowed_params for param in sig.parameters.keys())
        if not all_params_allowed:
            return None

        has_df_param = ("df" in sig.parameters) or ("dfs" in sig.parameters)
        has_only_df_param = has_df_param and (len(sig.parameters) == 1)

        return_annotation = sig.return_annotation

        if return_annotation is inspect.Signature.empty:
            has_return_annotation = False
            has_callable_return = False
        elif isinstance(return_annotation, str):
            has_return_annotation = True
            has_callable_return = return_annotation.startswith("Callable[")
        elif hasattr(return_annotation, "__name__"):
            has_return_annotation = return_annotation.__name__ != "_empty"
            has_callable_return = return_annotation.__name__.startswith("Callable")
        else:  # pragma: no cover
            raise ValueError(
                f"Cannot parse return annotation type for {compute_fn.__name__}: {return_annotation}"
            )

        if has_only_df_param:
            if has_return_annotation:
                return cls.DIRECT if not has_callable_return else None
            return cls.DIRECT
        elif has_df_param:
            if has_return_annotation:
                return cls.UNBOUND if not has_callable_return else None
            return cls.UNBOUND
        else:
            if has_return_annotation:
                return cls.FUNCTOR if has_callable_return else None
            return cls.FUNCTOR

        print(sig.return_annotation)


def identity_fn(df: Any) -> Any:
    """A "null" compute function that returns the input DataFrame as is.

    Args:
        df: The input DataFrame.

    Returns:
        The input DataFrame.

    Examples:
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> (identity_fn(df) == df).select(pl.all_horizontal(pl.all().all())).item()
        True
    """

    return df


def bind_compute_fn(cfg: DictConfig, stage_cfg: DictConfig, compute_fn: ANY_COMPUTE_FN_T) -> COMPUTE_FN_T:
    """Bind the compute function to the appropriate parameters based on the type of the compute function.

    Args:
        cfg: The DictConfig configuration object.
        stage_cfg: The DictConfig stage configuration object. This is separated from the ``cfg`` argument
            because in some cases, such as under the match & revise paradigm, the stage config may be modified
            dynamically under different matcher conditions to yield different compute functions.
        compute_fn: The compute function to bind.

    Returns:
        The compute function bound to the appropriate parameters.

    Raises:
        ValueError: If the compute function is not a valid compute function.

    Examples:
        >>> compute_fn = bind_compute_fn(DictConfig({}), DictConfig({}), None)
        >>> compute_fn("foobar")
        'foobar'
        >>> def compute_fntr(df: pl.DataFrame, cfg: DictConfig) -> pl.DataFrame:
        ...     return df.with_columns(pl.lit(cfg.val).alias("val"))
        >>> compute_fn = bind_compute_fn(DictConfig({"val": "foo"}), None, compute_fntr)
        >>> compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ val │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ foo │
        │ 2   ┆ foo │
        │ 3   ┆ foo │
        └─────┴─────┘
        >>> def direct_compute_fn(df: pl.DataFrame) -> pl.DataFrame:
        ...     return df.with_columns(pl.lit("bar").alias("val"))
        >>> compute_fn = bind_compute_fn(DictConfig({"val": "foo"}), None, direct_compute_fn)
        >>> compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ val │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ bar │
        │ 2   ┆ bar │
        │ 3   ┆ bar │
        └─────┴─────┘
        >>> def compute_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
        ...     return lambda df: df.with_columns(pl.lit(cfg.val).alias("val"))
        >>> compute_fn = bind_compute_fn(DictConfig({"val": "foo"}), None, compute_fntr)
        >>> compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ val │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ foo │
        │ 2   ┆ foo │
        │ 3   ┆ foo │
        └─────┴─────┘
        >>> def compute_fntr(stage_cfg, cfg) -> Callable[[pl.DataFrame], pl.DataFrame]:
        ...     return lambda df: df.with_columns(
        ...         pl.lit(stage_cfg.val).alias("stage_val"), pl.lit(cfg.val).alias("cfg_val")
        ...     )
        >>> compute_fn = bind_compute_fn(DictConfig({"val": "quo"}), DictConfig({"val": "bar"}), compute_fntr)
        >>> compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 3)
        ┌─────┬───────────┬─────────┐
        │ a   ┆ stage_val ┆ cfg_val │
        │ --- ┆ ---       ┆ ---     │
        │ i64 ┆ str       ┆ str     │
        ╞═════╪═══════════╪═════════╡
        │ 1   ┆ bar       ┆ quo     │
        │ 2   ┆ bar       ┆ quo     │
        │ 3   ┆ bar       ┆ quo     │
        └─────┴───────────┴─────────┘
        >>> def compute_fntr(df, code_metadata):
        ...     return df.join(code_metadata, on="a")
        >>> code_metadata_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmpdir:
        ...     code_metadata_fp = Path(tmpdir) / "codes.parquet"
        ...     code_metadata_df.write_parquet(code_metadata_fp)
        ...     stage_cfg = DictConfig({"metadata_input_dir": tmpdir})
        ...     compute_fn = bind_compute_fn(DictConfig({}), stage_cfg, compute_fntr)
        ...     compute_fn(pl.DataFrame({"a": [1, 2, 3]}))
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> def compute_fntr(df: pl.DataFrame, cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
        ...     return lambda df: df
        >>> bind_compute_fn(DictConfig({}), DictConfig({}), compute_fntr)
        Traceback (most recent call last):
            ...
        ValueError: Invalid compute function
    """

    if compute_fn is None:
        return identity_fn

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

    match ComputeFnType.from_fn(compute_fn):
        case ComputeFnType.DIRECT:
            return compute_fn
        case ComputeFnType.UNBOUND:
            return partial(compute_fn, **fntr_params(compute_fn))
        case ComputeFnType.FUNCTOR:
            return compute_fn(**fntr_params(compute_fn))
        case _:
            raise ValueError("Invalid compute function")
