from importlib.resources import files
import logging
import sys

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from . import __package_name__, __version__
from .configs import PipelineConfig
from .stages.discovery import get_all_registered_stages

logger = logging.getLogger(__name__)

HELP_STRS = {"--help", "-h", "help", "h"}
MAIN_CFG_PATH = files(__package_name__) / "configs" / "_main.yaml"


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


def run_stage():
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

    # Register the stage structured config and pipeline configuration
    stage = pipeline_cfg.register_for(stage_name)

    sys.argv = sys.argv[2:]  # remove dispatcher arguments

    cs = ConfigStore.instance()
    cs.store(name="_main", node=OmegaConf.load(MAIN_CFG_PATH))

    hydra_wrapper = hydra.main(version_base=None, config_name="_main")

    OmegaConf.register_new_resolver("get_package_version", lambda: __version__, replace=False)
    OmegaConf.register_new_resolver("get_package_name", lambda: __package_name__, replace=False)
    OmegaConf.register_new_resolver("stage_name", lambda: stage_name)
    OmegaConf.register_new_resolver("stage_docstring", lambda: stage.stage_docstring.replace("$", "$$"))

    hydra_wrapper(stage.main)()
