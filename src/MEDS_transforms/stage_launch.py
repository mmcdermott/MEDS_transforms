"""Set up and run a single stage, either via Hydra's CLI (`MEDS_transform-stage`) or in the calling process.

Both paths compose the stage's config the same way: the stage's resolved config is registered as the
``_pipeline`` node, ``configs/_main.yaml`` as ``_main``, and the resolvers it interpolates are registered for
this stage. `MEDS_transform-stage` then hands ``_main`` to ``hydra.main``, while `run_stage_in_process`
composes it with ``hydra.compose`` and calls the stage directly, which is what the pipeline runner's
``--in_process`` mode does for every stage.
"""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from . import __package_name__, __version__
from .configs import PipelineConfig
from .stage_outputs import carry_forward_unchanged_outputs

if TYPE_CHECKING:
    from .stages import Stage

logger = logging.getLogger(__name__)

MAIN_CFG_PATH = files(__package_name__) / "configs" / "_main.yaml"


def stamp_run_id(argv: list[str], run_id: str) -> list[str]:
    """Add a `run_id` override to `argv` unless the caller already set one.

    This runs once in the dispatcher process, before Hydra's `--multirun` sweep fans out, so every worker
    of the invocation inherits the same value and every fresh invocation gets a new one. That is what
    lets `do_overwrite=True` tell a sibling worker's just-written output apart from a previous run's
    leftovers; see `MEDS_transforms.mapreduce.rwlock.run_marker_dir`.

    Args:
        argv: The Hydra argument list, with the dispatcher's own arguments already stripped.
        run_id: The identifier to stamp.

    Hydra parses its overrides as a single positional argument, so they must be contiguous: an override
    after a flag such as `--multirun` or `--cfg job` that follows other overrides is rejected. The stamp is
    therefore put just before the caller's first override (or right after the program name, if there are
    none), never at the end.

    Returns:
        `argv`, with a `run_id` override added if it did not already carry one.

    Examples:
        >>> stamp_run_id(["some_stage", "worker=0"], "abc123")
        ['some_stage', 'run_id=abc123', 'worker=0']

        Flags before, between or after the overrides stay where they are:

        >>> stamp_run_id(["some_stage", "stage=x", 'worker="range(0,2)"', "--multirun"], "abc123")
        ['some_stage', 'run_id=abc123', 'stage=x', 'worker="range(0,2)"', '--multirun']
        >>> stamp_run_id(["some_stage", "--multirun", "stage=x", "--cfg", "job"], "abc123")
        ['some_stage', '--multirun', 'run_id=abc123', 'stage=x', '--cfg', 'job']
        >>> stamp_run_id(["some_stage", "--cfg", "job"], "abc123")
        ['some_stage', 'run_id=abc123', '--cfg', 'job']

        A caller that sets `run_id` itself — a wrapping tool coordinating several invocations, say —
        keeps their value, however it is spelled:

        >>> stamp_run_id(["some_stage", "run_id=mine"], "abc123")
        ['some_stage', 'run_id=mine']
        >>> stamp_run_id(["some_stage", "++run_id=mine"], "abc123")
        ['some_stage', '++run_id=mine']
        >>> stamp_run_id(["some_stage", "~run_id"], "abc123")
        ['some_stage', '~run_id']

        Overrides that merely start with the same characters are not mistaken for it:

        >>> stamp_run_id(["some_stage", "run_id_prefix=x"], "abc123")
        ['some_stage', 'run_id=abc123', 'run_id_prefix=x']
    """

    for arg in argv:
        key = arg.split("=", 1)[0].lstrip("+~")
        if key == "run_id":
            return argv

    first_override = next(
        (i for i, arg in enumerate(argv[1:], start=1) if "=" in arg and not arg.startswith("-")), 1
    )
    return [*argv[:first_override], f"run_id={run_id}", *argv[first_override:]]


def register_stage(pipeline_cfg: PipelineConfig, stage_name: str) -> Stage:
    """Register the configs and resolvers that composing `stage_name`'s config needs, and return the stage.

    Registering again replaces the previous stage's registrations, so one process can set up several
    stages in turn, as the in-process pipeline runner does.

    Args:
        pipeline_cfg: The pipeline the stage belongs to.
        stage_name: The stage to set up.

    Returns:
        The loaded stage.

    Examples:
        >>> pipeline_cfg = PipelineConfig(stages=["filter_subjects", "reorder_measurements"])
        >>> stage = register_stage(pipeline_cfg, "filter_subjects")
        >>> stage.stage_name
        'filter_subjects'
        >>> OmegaConf.create({"s": "${stage_name:}"}).s
        'filter_subjects'
        >>> _ = register_stage(pipeline_cfg, "reorder_measurements")
        >>> OmegaConf.create({"s": "${stage_name:}"}).s
        'reorder_measurements'
    """
    stage = pipeline_cfg.register_for(stage_name)

    ConfigStore.instance().store(name="_main", node=OmegaConf.load(MAIN_CFG_PATH))

    OmegaConf.register_new_resolver("get_package_version", lambda: __version__, replace=True)
    OmegaConf.register_new_resolver("get_package_name", lambda: __package_name__, replace=True)
    OmegaConf.register_new_resolver("stage_name", lambda: stage_name, replace=True)
    OmegaConf.register_new_resolver(
        "stage_docstring", lambda: (stage.stage_docstring or "").replace("$", "$$"), replace=True
    )
    return stage


def compose_stage_config(overrides: list[str]) -> DictConfig:
    """Compose the config of the most recently registered stage, as ``hydra.main`` would for one run.

    Args:
        overrides: Hydra overrides, as they would be passed on the command line (unquoted).

    Returns:
        The composed config, without the ``hydra`` node.

    Examples:
        >>> _ = register_stage(PipelineConfig(stages=["filter_subjects"]), "filter_subjects")
        >>> cfg = compose_stage_config(["output_dir=out", "stage_output_dir=${output_dir}/filter_subjects"])
        >>> print(cfg.stage, cfg.stage_cfg.data_output_dir)
        filter_subjects out/filter_subjects/data
        >>> "hydra" in cfg
        False
    """
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base=None):
        return compose(config_name="_main", overrides=overrides)


def run_stage_in_process(pipeline_config_fp: str | Path, stage_name: str, overrides: list[str]) -> None:
    """Run one stage in the calling process, as ``MEDS_transform-stage <pipeline> <stage> ...`` would.

    The stage runs as a single worker, then finishes its outputs (see `carry_forward_unchanged_outputs`)
    just as the stage launcher does once its workers are done. Its composed config and overrides are saved
    to ``<log_dir>/.hydra/``, and its log to ``<log_dir>/<stage>.log``, as a Hydra run would. Concurrent
    launches of the same pipeline, in process or not, still share the stage's work.

    Args:
        pipeline_config_fp: The pipeline YAML, as a path or in ``pkg://`` syntax.
        stage_name: The stage to run.
        overrides: Hydra overrides, unquoted. A fresh ``run_id`` is added unless one is given.

    Examples:
        >>> from pretty_print_directory import PrintConfig
        >>> from MEDS_transforms.stages.discovery import get_all_registered_stages
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     root = Path(tmpdir)
        ...     input_dir, output_dir = root / "input", root / "output"
        ...     input_dir.mkdir()
        ...     stage = get_all_registered_stages()["filter_subjects"].load()
        ...     stage.test_cases["."].write_for_test(input_dir)
        ...     pipeline_fp = root / "pipeline.yaml"
        ...     _ = pipeline_fp.write_text(
        ...         f"input_dir: {input_dir}\\noutput_dir: {output_dir}\\n"
        ...         "stages:\\n  - filter_subjects:\\n      min_events_per_subject: 5\\n"
        ...     )
        ...     run_stage_in_process(pipeline_fp, "filter_subjects", ["stage=filter_subjects"])
        ...     print_directory(output_dir, PrintConfig(ignore_regex=r"\\.lock$"))
        ├── .logs
        │   ├── .hydra
        │   │   ├── config.yaml
        │   │   └── overrides.yaml
        │   └── filter_subjects.log
        ├── data
        │   ├── held_out
        │   │   └── 0.parquet
        │   ├── train
        │   │   ├── 0.parquet
        │   │   └── 1.parquet
        │   └── tuning
        │       └── 0.parquet
        └── metadata
            ├── codes.parquet
            ├── dataset.json
            └── subject_splits.parquet
    """
    overrides = stamp_run_id([stage_name, *overrides], uuid4().hex)[1:]

    stage = register_stage(PipelineConfig.from_arg(pipeline_config_fp), stage_name)
    cfg = compose_stage_config(overrides)

    log_dir = Path(cfg.log_dir)
    (log_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, log_dir / ".hydra" / "config.yaml")
    OmegaConf.save(OmegaConf.create(overrides), log_dir / ".hydra" / "overrides.yaml")

    handler = logging.FileHandler(log_dir / f"{stage_name}.log")
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        logger.info(f"Running stage {stage_name} in process")
        stage.main(cfg)
        carry_forward_unchanged_outputs(cfg, stage.stage_type, stage.is_metadata)
    finally:
        root_logger.removeHandler(handler)
        handler.close()
