"""This script is a helper utility to run entire pipelines from a single script.

To do this effectively, this runner functionally takes a "meta configuration" file that contains:

  1. The path to the pipeline configuration file.
  2. Configuration details for how to run each stage of the pipeline, including mappings to the underlying
     stage scripts and Hydra launcher configurations for each stage to control parallelism, resources, etc.
"""

import argparse
import logging
import subprocess
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from .configs import PipelineConfig
from .configs.utils import OmegaConfResolver

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

logger = logging.getLogger(__name__)


def get_parallelization_args(
    parallelization_cfg: dict | DictConfig | None, default_parallelization_cfg: dict | DictConfig
) -> list[str]:
    """Extracts the specific parallelization arguments given the default and stage-specific configurations.

    Args:
        parallelization_cfg: The stage-specific parallelization configuration.
        default_parallelization_cfg: The default parallelization configuration.

    Returns:
        A list of command-line arguments for parallelization.

    Examples:
        >>> get_parallelization_args({}, {})
        []
        >>> get_parallelization_args(None, {"n_workers": 4})
        []
        >>> get_parallelization_args({"launcher": "joblib"}, {})
        ['--multirun', 'worker="range(0,1)"', 'hydra/launcher=joblib']
        >>> get_parallelization_args({"n_workers": 2, "launcher_params": "foo"}, {})
        Traceback (most recent call last):
            ...
        ValueError: If launcher_params is provided, launcher must also be provided.
        >>> get_parallelization_args({"n_workers": 2}, {})
        ['--multirun', 'worker="range(0,2)"']
        >>> get_parallelization_args({"launcher": "slurm"}, {"n_workers": 3, "launcher": "joblib"})
        ['--multirun', 'worker="range(0,3)"', 'hydra/launcher=slurm']
        >>> get_parallelization_args(
        ...     {"n_workers": 2, "launcher": "joblib"},
        ...     {"n_workers": 5, "launcher_params": {"foo": "bar"}},
        ... )
        ['--multirun', 'worker="range(0,2)"', 'hydra/launcher=joblib', 'hydra.launcher.foo=bar']
        >>> get_parallelization_args(
        ...     {"n_workers": 5, "launcher_params": {"biz": "baz"}, "launcher": "slurm"}, {}
        ... )
        ['--multirun', 'worker="range(0,5)"', 'hydra/launcher=slurm', 'hydra.launcher.biz=baz']
    """

    if parallelization_cfg is None:
        return []

    if len(parallelization_cfg) == 0 and len(default_parallelization_cfg) == 0:
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
        if "launcher_params" in parallelization_cfg:
            raise ValueError("If launcher_params is provided, launcher must also be provided.")

        return parallelization_args

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


def run_stage(
    pipeline_config_fp: str,
    stage_runners_cfg: dict | DictConfig,
    pipeline_cfg: PipelineConfig,
    stage_name: str,
    additional_args: list[str] | None = None,
    default_parallelization_cfg: dict | DictConfig | None = None,
    do_profile: bool = False,
    runner_fn: callable = subprocess.run,  # For dependency injection
):
    """Runs a single stage of the pipeline.

    Args:
        pipeline_config_fp: Path to the pipeline configuration on disk.
        stage_runners_cfg: The dictionary of stage runner configurations.
        stage_name: The name of the stage to run.
        additional_args: Additional command line arguments to forward to the
            stage invocation.
        default_parallelization_cfg: Default parallelization configuration.
        do_profile: Whether to enable Hydra profiling for the stage.
        runner_fn: Function used to actually run the stage command. This is
            primarily for dependency injection in testing.

    Raises:
        ValueError: If the stage fails to run.

    Examples:
        >>> def fake_shell_succeed(cmd, shell, capture_output):
        ...     print(cmd)
        ...     return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
        >>> def fake_shell_fail(cmd, shell, capture_output):
        ...     print(cmd)
        ...     return subprocess.CompletedProcess(args=cmd, returncode=1, stdout=b"", stderr=b"")
        >>> stage_runners = {
        ...     "reshard_to_split": {"_script": "not used"},
        ...     "fit_vocabulary_indices": {},
        ...     "reorder_measurements": {"script": "baz_script"},
        ... }
        >>> pipeline_cfg = PipelineConfig(
        ...     stages=[
        ...         "reshard_to_split",
        ...         {"fit_vocabulary_indices": {"_script": "foobar"}},
        ...         "reorder_measurements",
        ...     ],
        ... )
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reshard_to_split",
        ...     runner_fn=fake_shell_succeed,
        ... )
        MEDS_transform-stage pipeline_config.yaml reshard_to_split stage=reshard_to_split
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "fit_vocabulary_indices",
        ...     runner_fn=fake_shell_succeed,
        ... )
        foobar stage=fit_vocabulary_indices
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reorder_measurements",
        ...     runner_fn=fake_shell_succeed,
        ... )
        baz_script stage=reorder_measurements
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reorder_measurements",
        ...     do_profile=True,
        ...     runner_fn=fake_shell_succeed,
        ... )
        baz_script stage=reorder_measurements
            ++hydra.callbacks.profiler._target_=hydra_profiler.profiler.ProfilerCallback
        >>> stage_runners["reorder_measurements"]["parallelize"] = {"n_workers": 2}
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reorder_measurements",
        ...     runner_fn=fake_shell_succeed,
        ... )
        baz_script --multirun stage=reorder_measurements worker="range(0,2)"
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reorder_measurements",
        ...     runner_fn=fake_shell_fail,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Stage reorder_measurements failed via ...

        Improper configurations also raise errors:

        >>> bad_runners = {"reshard_to_split": {"_base_stage": "belongs in the stage"}}
        >>> bad_cfg = PipelineConfig(stages=["reshard_to_split"])
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     bad_runners,
        ...     bad_cfg,
        ...     "reshard_to_split",
        ...     runner_fn=fake_shell_succeed,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Put _base_stage args is in your pipeline config
    """

    if default_parallelization_cfg is None:
        default_parallelization_cfg = {}

    stage_config = pipeline_cfg.parsed_stages_by_name[stage_name].config
    stage_runner_config = stage_runners_cfg.get(stage_name, {})

    script = None
    if "script" in stage_runner_config:
        script = stage_runner_config.get("script")
    elif "_script" in stage_config:
        script = stage_config.get("_script")
    elif "_base_stage" in stage_runner_config:
        raise ValueError("Put _base_stage args is in your pipeline config")
    else:
        script = f"MEDS_transform-stage {pipeline_config_fp} {stage_name}"

    command_parts = [
        script,
        f"stage={stage_name}",
    ]

    parallelization_args = get_parallelization_args(
        stage_runner_config.get("parallelize", {}), default_parallelization_cfg
    )

    if parallelization_args:
        multirun = parallelization_args.pop(0)
        command_parts = command_parts[:1] + [multirun] + command_parts[1:] + parallelization_args

    if additional_args:
        command_parts.extend(additional_args)

    if do_profile:
        command_parts.append("++hydra.callbacks.profiler._target_=hydra_profiler.profiler.ProfilerCallback")

    full_cmd = " ".join(command_parts)
    logger.info(f"Running command: {full_cmd}")
    command_out = runner_fn(full_cmd, shell=True, capture_output=True)

    # https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging

    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    logger.info(f"Command output:\n{stdout}")
    logger.info(f"Command error:\n{stderr}")

    if command_out.returncode != 0:
        raise ValueError(
            f"Stage {stage_name} failed via {full_cmd} with return code {command_out.returncode}."
        )


def main(argv: list[str] | None = None) -> int:
    """Run an entire pipeline based on command line arguments."""

    parser = argparse.ArgumentParser(description="MEDS-Transforms Pipeline Runner")
    parser.add_argument("pipeline_config_fp")
    parser.add_argument("stage_runner_fp", nargs="?", default=None)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    pipeline_config = PipelineConfig.from_arg(args.pipeline_config_fp)
    stage_runners_cfg = load_yaml_file(args.stage_runner_fp)

    stages = [s.name for s in pipeline_config.parsed_stages]
    if not stages:
        raise ValueError("Pipeline configuration must specify at least one stage.")

    log_dir = Path(pipeline_config.additional_params["output_dir"]) / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    global_done_file = log_dir / "_all_stages.done"
    if global_done_file.exists():
        logger.info("All stages are already complete. Exiting.")
        return 0

    if "parallelize" in stage_runners_cfg:
        default_parallelization_cfg = stage_runners_cfg["parallelize"]
    elif "parallelize" in pipeline_config.additional_params:
        default_parallelization_cfg = pipeline_config.additional_params["parallelize"]
    else:
        default_parallelization_cfg = None

    for stage in stages:
        done_file = log_dir / f"{stage}.done"

        if done_file.exists():
            logger.info(f"Skipping stage {stage} as it is already complete.")
        else:
            logger.info(f"Running stage: {stage}")
            run_stage(
                args.pipeline_config_fp,
                stage_runners_cfg,
                pipeline_config,
                stage,
                additional_args=args.overrides,
                default_parallelization_cfg=default_parallelization_cfg,
            )
            done_file.touch()

    global_done_file.touch()
    return 0


@OmegaConfResolver
def load_yaml_file(path: str | None) -> dict | DictConfig:
    """Loads a YAML file as an OmegaConf object.

    Args:
        path: The path to the YAML file.

    Returns:
        The OmegaConf object representing the YAML file, or None if no path is provided.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        >>> load_yaml_file(None)
        {}
        >>> load_yaml_file("nonexistent_file.yaml")
        Traceback (most recent call last):
            ...
        FileNotFoundError: File nonexistent_file.yaml does not exist.
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        ...     _ = f.write(b"foo: bar")
        ...     f.flush()
        ...     load_yaml_file(f.name)
        {'foo': 'bar'}
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        ...     cfg = OmegaConf.create({"foo": "bar"})
        ...     OmegaConf.save(cfg, f.name)
        ...     load_yaml_file(f.name)
        {'foo': 'bar'}
    """

    if not path:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    try:
        return OmegaConf.load(path)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to load {path} as an OmegaConf: {e}. Trying as a plain YAML file.")
        yaml_text = path.read_text()
        return yaml.load(yaml_text, Loader=Loader)
