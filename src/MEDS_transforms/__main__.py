import logging
import sys
from importlib.resources import files
from uuid import uuid4

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from . import __package_name__, __version__
from .configs import PipelineConfig
from .stages.discovery import get_all_registered_stages

logger = logging.getLogger(__name__)

HELP_STRS = {"--help", "-h", "help", "h"}
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

    Returns:
        `argv`, with a `run_id` override appended if it did not already carry one.

    Examples:
        >>> stamp_run_id(["some_stage", "worker=0"], "abc123")
        ['some_stage', 'worker=0', '++run_id=abc123']

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
        ['some_stage', 'run_id_prefix=x', '++run_id=abc123']
    """

    for arg in argv:
        key = arg.split("=", 1)[0].lstrip("+~")
        if key == "run_id":
            return argv

    return [*argv, f"++run_id={run_id}"]


def print_help_stage():
    """Print help for all stages."""

    all_stage_names = list(get_all_registered_stages().keys())

    print(f"Usage: {sys.argv[0]} <pipeline_yaml> <stage_name> [args]")
    print(
        "  * pipeline_yaml: Path to the pipeline YAML file on disk or in the "
        "'pkg://<pkg_name>.<relative_path>' format."
    )
    print("  * stage_name: Name of the stage to run.")
    print()
    print("Available stages:")
    for name in sorted(all_stage_names):
        print(f"  - {name}")


def run_stage():  # pragma: no cover
    """Run a stage based on command line arguments."""

    if len(sys.argv) < 2:
        print_help_stage()
        sys.exit(1)
    elif sys.argv[1] in HELP_STRS:
        print_help_stage()
        sys.exit(0)
    elif len(sys.argv) < 3:
        print_help_stage()
        sys.exit(1)

    pipeline_cfg = PipelineConfig.from_arg(sys.argv[1])
    stage_name = sys.argv[2]

    sys.argv = sys.argv[2:]  # remove dispatcher arguments

    # Stamped here, in the single dispatcher process, so that a `--multirun` sweep hands the same value
    # to every worker it launches.
    sys.argv = stamp_run_id(sys.argv, uuid4().hex)

    # Register the stage structured config and pipeline configuration
    stage = pipeline_cfg.register_for(stage_name)

    cs = ConfigStore.instance()
    cs.store(name="_main", node=OmegaConf.load(MAIN_CFG_PATH))

    hydra_wrapper = hydra.main(version_base=None, config_name="_main")

    OmegaConf.register_new_resolver("get_package_version", lambda: __version__, replace=False)
    OmegaConf.register_new_resolver("get_package_name", lambda: __package_name__, replace=False)
    OmegaConf.register_new_resolver("stage_name", lambda: stage_name)
    OmegaConf.register_new_resolver("stage_docstring", lambda: stage.stage_docstring.replace("$", "$$"))

    hydra_wrapper(stage.main)()
