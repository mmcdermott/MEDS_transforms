"""Functions for registering and defining MEDS-transforms stages."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .configs import PREPROCESS_CONFIG_YAML
from .mapreduce import ANY_COMPUTE_FN_T, map_stage, mapreduce_stage

logger = logging.getLogger(__name__)

MAIN_FN_T = Callable[[DictConfig], None]


class StageType(StrEnum):
    """The types of stages MEDS-Transforms supports.

    Attributes:
        MAP: A stage that applies a transformation to each data shard of a MEDS dataset, outputting new data
            shards.
        MAPREDUCE: A stage that applies a metadata extraction operation to each data shard of a MEDS dataset,
            then reduces the outputs of those transformations into an updated metadata/codes.parquet file.
        MAIN: A stage that does not fit into either of the above categories, and provides a direct main
            function.
    """

    MAP = "map"
    MAPREDUCE = "mapreduce"
    MAIN = "main"

    @classmethod
    def from_fns(
        cls, main_fn: MAIN_FN_T | None, map_fn: ANY_COMPUTE_FN_T | None, reduce_fn: ANY_COMPUTE_FN_T | None
    ) -> StageType:
        """Determines the stage type based on the provided functions.

        Args:
            main_fn: The main function for the stage. May be None.
            map_fn: The mapping function for the stage. May be None.
            reduce_fn: The reducing function for the stage. May be None.

        Returns:
            StageType: The type of stage determined from the provided functions. If the passed functions do
                not correspond to a valid stage type, a ValueError will be raised.

        Raises:
            ValueError: If the provided functions do not correspond to a valid stage type.

        Examples:
            >>> def main_fn(cfg: DictConfig):
            ...     pass
            >>> def map_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     pass
            >>> def reduce_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     pass
            >>> StageType.from_fns(main_fn=main_fn, map_fn=None, reduce_fn=None)
            <StageType.MAIN: 'main'>
            >>> StageType.from_fns(main_fn=None, map_fn=map_fn, reduce_fn=None)
            <StageType.MAP: 'map'>
            >>> StageType.from_fns(main_fn=None, map_fn=map_fn, reduce_fn=reduce_fn)
            <StageType.MAPREDUCE: 'mapreduce'>
            >>> StageType.from_fns(main_fn=None, map_fn=None, reduce_fn=None)
            Traceback (most recent call last):
                ...
            ValueError: Either main_fn or map_fn/reduce_fn must be provided.
            >>> StageType.from_fns(main_fn=main_fn, map_fn=map_fn, reduce_fn=reduce_fn)
            Traceback (most recent call last):
                ...
            ValueError: Only one of main_fn or map_fn/reduce_fn should be provided.
        """

        if main_fn is not None:
            if map_fn is not None or reduce_fn is not None:
                raise ValueError("Only one of main_fn or map_fn/reduce_fn should be provided.")
            return StageType.MAIN
        elif map_fn is not None and reduce_fn is not None:
            return StageType.MAPREDUCE
        elif map_fn is not None:
            return StageType.MAP
        else:
            raise ValueError("Either main_fn or map_fn/reduce_fn must be provided.")


def get_stage_main(
    *,
    main_fn: MAIN_FN_T | None = None,
    map_fn: ANY_COMPUTE_FN_T | None = None,
    reduce_fn: ANY_COMPUTE_FN_T | None = None,
    config_path: Path | None = None,
    stage_name: str | None = None,
    stage_docstring: str | None = None,
) -> MAIN_FN_T:
    """Wraps or returns a function that can serve as the main function for a stage."""

    mode = StageType.from_fns(main_fn, map_fn, reduce_fn)

    if config_path is None:
        config_path = PREPROCESS_CONFIG_YAML

    hydra_wrapper = hydra.main(
        version_base=None,
        config_path=str(config_path.parent),
        config_name=config_path.stem,
    )

    if stage_name is None:
        stage_name = (main_fn or map_fn).__module__.split(".")[-1]

    match mode:
        case StageType.MAIN:
            stage_docstring = stage_docstring or inspect.getdoc(main_fn) or ""
        case StageType.MAP:
            stage_docstring = stage_docstring or inspect.getdoc(map_fn) or ""

            @functools.wraps(map_fn)
            def main_fn(cfg: DictConfig):
                return map_stage(cfg, map_fn)

        case StageType.MAPREDUCE:
            stage_docstring = stage_docstring or (
                f"Map Stage:\n{inspect.getdoc(map_fn) or ''}\n\n"
                f"Reduce stage:\n{inspect.getdoc(reduce_fn) or ''}"
            )

            def main_fn(cfg: DictConfig):
                return mapreduce_stage(cfg, map_fn, reduce_fn)

    main_fn.__name__ = stage_name
    main_fn.__doc__ = stage_docstring

    hydra_wraped_main = hydra_wrapper(main_fn)

    @functools.wraps(hydra_wraped_main)
    def wrapped_main(*args, **kwargs):
        OmegaConf.register_new_resolver("current_script_name", lambda: stage_name, replace=True)
        OmegaConf.register_new_resolver(
            "get_script_docstring", lambda: stage_docstring.replace("$", "$$"), replace=True
        )
        return hydra_wraped_main(*args, **kwargs)

    return wrapped_main


def MEDS_transforms_stage(*args, **kwargs):
    """This is a decorator used to define and register a MEDS-Transforms stage of any variety.

    It can be used in a variety of ways, depending on the arguments provided and the manner of invocation:

      1. If used as a decorator without arguments to a single function named `main`, it will define a
         `StageType.MAIN` stage using that function, and the return value will be the target executable main
         function itself (which will have an identical signature to the passed function). The function will
         not have other methods registered to it.
      2. If used as a decorator without arguments to a single function _not_ named `main`, it will define a
         `StageType.MAP` stage using that function or functor as the mapper. The return value will be a new
         function object that has the same signature as the passed function, but with a new method `main`
         attached to it that contains the main CLI entry point for the stage and can be invoked with hydra.
      3. If used as a decorator with keyword arguments to a single function named `main`, and the arguments do
         not include `map_fn` or `reduce_fn`, it will define a `StageType.MAIN` stage using that function, and
         the other keyword arguments will overwrite the defaults used in the definition of the stage.
      4. If used as a decorator with keyword arguments to a single function _not_ named `main`, and the
         arguments do not include `map_fn` or `reduce_fn`, it will define a `StageType.MAP` stage using that
         function or functor as the mapper, and the other keyword arguments will overwrite the defaults used
         in the definition of that stage.
      5. If used as a decorator with keyword arguments to a single function regardless of the name, and the
         keyword arguments define a `reduce_fn` but do not define a `map_fn`, then a `StageType.MAPREDUCE`
         stage will be created and attached to the function via the `main` entry point, with the passed
         function treated as the mapper and the `reduce_fn` as the reducer. The other keyword arguments will
         be used as normal. The function decorated will be returned without alteration to its signature.
      6. If used as a decorator to a single function with the `map_fn` keyword argument provided, it will
         throw an error.
      7. If called directly with only keyword arguments, not as a decorator to a function, with either
         `main_fn`, `map_fn`, or both `map_fn` and `reduce_fn` defined, it will produce the associated stage
         for those settings and return the main function directly, without modifying any passed function.
      8. If called with multiple position arguments, it will throw an error.
    """

    if len(args) > 1:
        raise ValueError(
            f"MEDS_transforms_stage can only be used with at most a single positional arg. Got {len(args)}"
        )
    if len(args) == 0:
        return get_stage_main(**kwargs)

    fn = args[0]
    if not inspect.isfunction(fn):
        raise TypeError(f"First argument must be a function. Got {type(fn)}")

    if "main_fn" in kwargs or "map_fn" in kwargs:
        raise ValueError("Cannot provide main_fn or map_fn when using as a decorator.")

    if fn.__name__ == "main" and "reduce_fn" not in kwargs:
        return get_stage_main(main_fn=fn, **kwargs)

    main_fn = get_stage_main(main_fn=None, map_fn=fn, **kwargs)
    fn.main = main_fn
    return fn
