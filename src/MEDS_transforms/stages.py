"""Functions for registering and defining MEDS-transforms stages."""

import functools
import inspect
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms.configs import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce import ANY_COMPUTE_FN_T, map_stage, mapreduce_stage

logger = logging.getLogger(__name__)


def registered_stage(
    compute_fn: ANY_COMPUTE_FN_T,
    stage_name: str | None = None,
    stage_docstring: str | None = None,
    reduce_fn: ANY_COMPUTE_FN_T | None = None,
    config_path: Path | None = None,
):
    """Wraps or returns a function that can serve as the main function for a stage."""

    if config_path is None:
        config_path = PREPROCESS_CONFIG_YAML

    if stage_name is None:
        stage_name = compute_fn.__module__.split(".")[-1]

    OmegaConf.register_new_resolver("current_script_name", lambda: stage_name, replace=True)

    if reduce_fn is None:
        if stage_docstring is None:
            stage_docstring = inspect.getdoc(compute_fn) or ""

        @functools.wraps(compute_fn)
        def main_fn(cfg: DictConfig):
            return map_stage(cfg, compute_fn)

    else:
        if stage_docstring is None:
            compute_fn_docstring = inspect.getdoc(compute_fn) or ""
            reduce_fn_docstring = inspect.getdoc(reduce_fn) or ""
            stage_docstring = f"Map Stage:\n{compute_fn_docstring}\n\nReduce stage:\n{reduce_fn_docstring}"

        def main_fn(cfg: DictConfig):
            return mapreduce_stage(cfg, compute_fn, reduce_fn)

    main_fn.__name__ = stage_name
    main_fn.__doc__ = stage_docstring

    # Replace $ with $$ in the docstring to avoid issues with OmegaConf
    stage_docstring = stage_docstring.replace("$", "$$")

    OmegaConf.register_new_resolver("get_script_docstring", lambda: stage_docstring, replace=True)

    return hydra.main(version_base=None, config_path=str(config_path.parent), config_name=config_path.stem)(
        main_fn
    )
