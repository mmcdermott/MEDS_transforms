import os
import sys
from collections.abc import Sequence
from importlib.resources import files

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from . import __package_name__, __version__
from .configs import PipelineConfig
from .utils import populate_stage

HELP_STRS = {"--help", "-h", "help", "h"}
MAIN_CFG_PATH = files(__package_name__) / "configs" / "_main.yaml"


def print_help_stage(all_stage_names: Sequence[str]):
    """Print help for all stages."""
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


def run_stage():
    """Run a stage based on command line arguments."""

    # We disable stage validation here as it is not needed on the CLI; instead, we manually validate that the
    # stage name matches after loading, and that plus the checks for duplicate stage entry points covers all
    # validation failure scenarios.
    os.environ["DISABLE_STAGE_VALIDATION"] = "1"

    from .stages import get_all_registered_stages

    all_stages = get_all_registered_stages()
    all_stage_names = list(all_stages.keys())

    if len(sys.argv) < 2:
        print_help_stage(all_stage_names)
        sys.exit(1)
    elif sys.argv[1] in HELP_STRS:
        print_help_stage(all_stage_names)
        sys.exit(0)
    elif len(sys.argv) < 3:
        print_help_stage(all_stage_names)
        sys.exit(1)

    pipeline_cfg = PipelineConfig.from_arg(sys.argv[1])
    stage_name = sys.argv[2]
    sys.argv = sys.argv[2:]  # remove dispatcher arguments

    if stage_name not in all_stages:
        raise ValueError(f"Stage '{stage_name}' not found.")

    stage = all_stages[stage_name].load()

    if stage.stage_name != stage_name:
        raise ValueError(
            f"Loaded stage name '{stage.stage_name}' does not match the provided name '{stage_name}'!"
        )

    _stage_configs = {}
    for ep in all_stages.values():
        _stage_configs.update(ep.load().default_config)

    cs = ConfigStore.instance()
    cs.store(group="stage_configs", name="_stage_configs", node=_stage_configs)
    cs.store(name="_pipeline", node=pipeline_cfg.structured_config)
    cs.store(name="_main", node=OmegaConf.load(MAIN_CFG_PATH))

    hydra_wrapper = hydra.main(version_base=None, config_name="_main")

    OmegaConf.register_new_resolver("populate_stage", populate_stage, replace=False)
    OmegaConf.register_new_resolver("get_package_version", lambda: __version__, replace=False)
    OmegaConf.register_new_resolver("get_package_name", lambda: __package_name__, replace=False)
    OmegaConf.register_new_resolver("stage_name", lambda: stage_name)
    OmegaConf.register_new_resolver("stage_docstring", lambda: stage.stage_docstring.replace("$", "$$"))

    hydra_wrapper(stage.main)()
