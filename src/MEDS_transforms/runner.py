#!/usr/bin/env python
"""This script is a helper utility to run entire pipelines from a single script.

To do this effectively, this runner functionally takes a "meta configuration" file that contains:
  1. The path to the pipeline configuration file.
  2. Configuration details for how to run each stage of the pipeline, including mappings to the underlying
     stage scripts and Hydra launcher configurations for each stage to control parallelism, resources, etc.
"""

import importlib
import subprocess
from pathlib import Path

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from MEDS_transforms import RESERVED_CONFIG_NAMES, RUNNER_CONFIG_YAML
from MEDS_transforms.utils import hydra_loguru_init


def get_script_from_name(stage_name: str) -> str | None:
    """Returns the script name for the given stage name.

    Args:
        stage_name: The name of the stage.

    Returns:
        The script name for the given stage name.
    """

    try:
        _ = importlib.import_module(f"MEDS_transforms.extract.{stage_name}")
        return f"MEDS_extract-{stage_name}"
    except ImportError:
        pass

    for pfx in ("MEDS_transforms.transforms", "MEDS_transforms.filters", "MEDS_transforms"):
        try:
            _ = importlib.import_module(f"{pfx}.{stage_name}")
            return f"MEDS_transform-{stage_name}"
        except ImportError:
            pass

    return None


def get_parallelization_args(
    parallelization_cfg: dict | DictConfig | None, default_parallelization_cfg: dict | DictConfig
) -> list[str]:
    """Gets the parallelization args."""

    if parallelization_cfg is None or len(parallelization_cfg) == 0:
        return []

    if "n_workers" in parallelization_cfg:
        n_workers = parallelization_cfg["n_workers"]
    elif "n_workers" in default_parallelization_cfg:
        n_workers = default_parallelization_cfg["n_workers"]
    else:
        n_workers = 1

    parallelization_args = [
        "--multirun",
        f'worker="range(0,{n_workers})"',
    ]

    if "launcher" in parallelization_cfg:
        launcher = parallelization_cfg["launcher"]
    elif "launcher" in default_parallelization_cfg:
        launcher = default_parallelization_cfg["launcher"]
    else:
        launcher = None

    if launcher is None:
        return parallelization_args

        if "launcher_params" in parallelization_cfg:
            raise ValueError("If launcher_params is provided, launcher must also be provided.")

    parallelization_args.append(f"hydra/launcher={launcher}")

    if "launcher_params" in parallelization_cfg:
        launcher_params = parallelization_cfg["launcher_params"]
    elif "launcher_params" in default_parallelization_cfg:
        launcher_params = default_parallelization_cfg["launcher_params"]
    else:
        launcher_params = {}

    for k, v in launcher_params.items():
        parallelization_args.append(f"hydra.launcher.{k}={v}")

    return parallelization_args


def run_stage(cfg: DictConfig, stage_name: str, default_parallelization_cfg: dict | DictConfig | None = None):
    """Runs a single stage of the pipeline.

    Args:
        cfg: The configuration for the entire pipeline.
        stage_name: The name of the stage to run.
    """

    if default_parallelization_cfg is None:
        default_parallelization_cfg = {}

    do_profile = cfg.get("do_profile", False)
    pipeline_config_fp = Path(cfg.pipeline_config_fp)
    stage_config = cfg._local_pipeline_config.stage_configs.get(stage_name, {})
    stage_runner_config = cfg._stage_runners.get(stage_name, {})

    script = None
    if "script" in stage_runner_config:
        script = stage_runner_config.script
    elif "_script" in stage_config:
        script = stage_config._script
    elif get_script_from_name(stage_name):
        script = get_script_from_name(stage_name)
    else:
        raise ValueError(f"Cannot determine script for {stage_name}")

    command_parts = [
        script,
        f"--config-dir={str(pipeline_config_fp.parent.resolve())}",
        f"--config-name={pipeline_config_fp.stem}",
        "'hydra.searchpath=[pkg://MEDS_transforms.configs]'",
        f"stage={stage_name}",
    ]

    command_parts.extend(
        get_parallelization_args(stage_runner_config.get("parallelize", {}), default_parallelization_cfg)
    )

    if do_profile:
        command_parts.append("++hydra.callbacks.profiler._target_=hydra_profiler.profiler.ProfilerCallback")

    full_cmd = " ".join(command_parts)
    logger.info(f"Running command: {full_cmd}")
    command_out = subprocess.run(full_cmd, shell=True, capture_output=True)

    # https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
    # https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.parse

    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    logger.info(f"Command output:\n{stdout}")
    logger.info(f"Command error:\n{stderr}")

    if command_out.returncode != 0:
        raise ValueError(
            f"Stage {stage_name} failed via {full_cmd} with return code {command_out.returncode}."
        )


@hydra.main(
    version_base=None, config_path=str(RUNNER_CONFIG_YAML.parent), config_name=RUNNER_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Runs the entire pipeline, end-to-end, based on the configuration provided.

    This script will launch many subsidiary commands via `subprocess`, one for each stage of the specified
    pipeline.
    """

    hydra_loguru_init()

    pipeline_config_fp = Path(cfg.pipeline_config_fp)
    if not pipeline_config_fp.exists():
        raise FileNotFoundError(f"Pipeline configuration file {pipeline_config_fp} does not exist.")
    if not pipeline_config_fp.suffix == ".yaml":
        raise ValueError(f"Pipeline configuration file {pipeline_config_fp} must have a .yaml extension.")
    if pipeline_config_fp.stem in RESERVED_CONFIG_NAMES:
        raise ValueError(
            f"Pipeline configuration file {pipeline_config_fp} must not have a name in "
            f"{RESERVED_CONFIG_NAMES}."
        )

    pipeline_config = load_yaml_file(cfg.pipeline_config_fp)
    stages = pipeline_config.get("stages", [])
    if not stages:
        raise ValueError("Pipeline configuration must specify at least one stage.")

    log_dir = Path(cfg.log_dir)

    if cfg.get("do_profile", False):
        try:
            pass
        except ImportError as e:
            raise ValueError(
                "You can't run in profiling mode without installing hydra-profiler. Try installing "
                "MEDS-transforms with the 'profiler' optional dependency: "
                "`pip install MEDS-transforms[profiler]`."
            ) from e

    global_done_file = log_dir / "_all_stages.done"
    if global_done_file.exists():
        logger.info("All stages are already complete. Exiting.")
        return

    if "parallelize" in cfg:
        default_parallelization_cfg = cfg.parallelize
    else:
        default_parallelization_cfg = None

    for stage in stages:
        done_file = log_dir / f"{stage}.done"

        if done_file.exists():
            logger.info(f"Skipping stage {stage} as it is already complete.")
        else:
            logger.info(f"Running stage: {stage}")
            run_stage(cfg, stage, default_parallelization_cfg=default_parallelization_cfg)
            done_file.touch()

    global_done_file.touch()


def load_yaml_file(path: str | None) -> dict | DictConfig:
    if not path:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    try:
        return OmegaConf.load(path)
    except Exception as e:
        logger.warning(f"Failed to load {path} as an OmegaConf: {e}. Trying as a plain YAML file.")
        yaml_text = path.read_text()
        return yaml.load(yaml_text, Loader=Loader)


def fix_str_for_path(s: str) -> str:
    """Replaces all space characters with underscores and all slashes with periods."""
    return s.replace(" ", "_").replace("/", ".")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("load_yaml_file", load_yaml_file, replace=False)
    OmegaConf.register_new_resolver("fix_str_for_path", fix_str_for_path, replace=False)

    main()
