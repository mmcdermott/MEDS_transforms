"""Functions for registering and defining MEDS-transforms stages."""

import functools
import inspect
import logging
from collections.abc import Callable
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .configs import PREPROCESS_CONFIG_YAML
from .mapreduce import ANY_COMPUTE_FN_T, map_stage, mapreduce_stage

logger = logging.getLogger(__name__)


def make_main_fn(
    compute_fn: ANY_COMPUTE_FN_T | None = None,
    reduce_fn: ANY_COMPUTE_FN_T | None = None,
) -> Callable[[DictConfig], None]:
    if compute_fn is None:
        raise ValueError("compute_fn must be provided")

    if reduce_fn is None:
        docstring = inspect.getdoc(compute_fn) or ""

        @functools.wraps(compute_fn)
        def main_fn(cfg: DictConfig):
            return map_stage(cfg, compute_fn)

    else:
        docstring = (
            f"Map Stage:\n{inspect.getdoc(compute_fn) or ''}\n\n"
            f"Reduce stage:\n{inspect.getdoc(reduce_fn) or ''}"
        )

        def main_fn(cfg: DictConfig):
            return mapreduce_stage(cfg, compute_fn, reduce_fn)

    main_fn.__doc__ = docstring
    return main_fn


def MEDS_transforms_stage(
    main_fn: Callable[[DictConfig], None] | None = None,
    compute_fn: ANY_COMPUTE_FN_T | None = None,
    reduce_fn: ANY_COMPUTE_FN_T | None = None,
    config_path: Path | None = None,
    stage_name: str | None = None,
    stage_docstring: str | None = None,
):
    """Wraps or returns a function that can serve as the main function for a stage."""

    if config_path is None:
        config_path = PREPROCESS_CONFIG_YAML

    hydra_wrapper = hydra.main(
        version_base=None,
        config_path=str(config_path.parent),
        config_name=config_path.stem,
    )

    if stage_name is None:
        stage_name = (main_fn or compute_fn).__module__.split(".")[-1]

    if main_fn is None:
        main_fn = make_main_fn(compute_fn, reduce_fn)
        main_fn.__name__ = stage_name

    if stage_docstring is None:
        stage_docstring = inspect.getdoc(main_fn) or ""

    hydra_wraped_main = hydra_wrapper(main_fn)

    @functools.wraps(hydra_wraped_main)
    def wrapped_main(*args, **kwargs):
        OmegaConf.register_new_resolver("current_script_name", lambda: stage_name, replace=True)
        OmegaConf.register_new_resolver(
            "get_script_docstring", lambda: stage_docstring.replace("$", "$$"), replace=True
        )
        return hydra_wraped_main(*args, **kwargs)

    return wrapped_main
