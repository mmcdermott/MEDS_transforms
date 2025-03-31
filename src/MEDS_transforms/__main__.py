import os
import sys
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from .configs import MAIN_YAML, register_structured_config
from .stages import get_all_registered_stages

HELP_STRS = {"--help", "-h", "help", "h"}
PKG_PFX = "pkg://"
YAML_EXTENSIONS = {"yaml", "yml"}
SINGLETON_STR = "__null__"


def print_help_stage():
    """Print help for all stages."""
    print(f"Usage: {sys.argv[0]} <pipeline_yaml> <stage_name> [args]")
    print(
        "  * pipeline_yaml: Path to the pipeline YAML file on disk or in the "
        f"'{PKG_PFX}<pkg_name>.<relative_path>' format."
    )
    print("  * stage_name: Name of the stage to run.")
    print()
    print("Available stages:")
    all_stages = get_all_registered_stages()
    for name in sorted(all_stages):
        print(f"  - {name}")


def resolve_pipeline_yaml(pipeline_yaml: str):
    """Resolve the pipeline YAML file path."""
    if pipeline_yaml == SINGLETON_STR:
        return None

    if pipeline_yaml.startswith(PKG_PFX):
        pipeline_yaml = pipeline_yaml[len(PKG_PFX) :]
        pipeline_parts = pipeline_yaml.split(".")

        if pipeline_parts[-1] not in YAML_EXTENSIONS:
            raise ValueError(
                f"Invalid pipeline YAML path '{pipeline_yaml}'. "
                f"Expected a file with one of the following extensions: {YAML_EXTENSIONS}"
            )

        pkg_name = pipeline_parts[0]
        suffix = pipeline_parts[-1]
        relative_path = Path(os.path.join(*pipeline_parts[1:-1])).with_suffix(f".{suffix}")

        try:
            pipeline_yaml = files(pkg_name) / relative_path
        except ImportError:
            raise ValueError(f"Package '{pkg_name}' not found. Please check the package name.")
    else:
        pipeline_yaml = Path(pipeline_yaml)

    if not pipeline_yaml.is_file():
        raise FileNotFoundError(f"Pipeline YAML file '{pipeline_yaml}' does not exist.")

    return pipeline_yaml


def run_stage():
    """Run a stage based on command line arguments."""

    all_stages = get_all_registered_stages()

    if len(sys.argv) < 2:
        print_help_stage()
        sys.exit(1)
    elif sys.argv[1] in HELP_STRS:
        print_help_stage()
        sys.exit(0)
    elif len(sys.argv) < 3:
        print_help_stage()
        sys.exit(1)

    pipeline_yaml = resolve_pipeline_yaml(sys.argv[1])
    stage_name = sys.argv[2]
    sys.argv = sys.argv[2:]  # remove dispatcher arguments

    executable_stage_name = register_structured_config(pipeline_yaml, stage_name)

    if executable_stage_name not in all_stages:
        raise ValueError(f"Stage '{executable_stage_name}' not found.")

    main_fn = all_stages[executable_stage_name]["entry_point"].load()

    OmegaConf.register_new_resolver("stage_name", lambda: stage_name)
    OmegaConf.register_new_resolver("stage_docstring", lambda: main_fn.__doc__.replace("$", "$$"))

    hydra_wrapper = hydra.main(
        version_base=None,
        config_path=str(MAIN_YAML.parent),
        config_name=MAIN_YAML.stem,
    )

    hydra_wrapper(main_fn)()
