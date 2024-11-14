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
except ImportError:  # pragma: no cover
    from yaml import Loader

from MEDS_transforms import RESERVED_CONFIG_NAMES, RUNNER_CONFIG_YAML
from MEDS_transforms.utils import hydra_loguru_init


def get_script_from_name(stage_name: str) -> str | None:
    """Returns the script name for the given stage name.

    Args:
        stage_name: The name of the stage.

    Returns:
        The script name for the given stage name.

    Examples:
        >>> get_script_from_name("shard_events")
        'MEDS_extract-shard_events'
        >>> get_script_from_name("fit_vocabulary_indices")
        'MEDS_transform-fit_vocabulary_indices'
        >>> get_script_from_name("filter_subjects")
        'MEDS_transform-filter_subjects'
        >>> get_script_from_name("reorder_measurements")
        'MEDS_transform-reorder_measurements'
        >>> get_script_from_name("nonexistent_stage")
        Traceback (most recent call last):
            ...
        ValueError: Could not find a script for stage nonexistent_stage.
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

    raise ValueError(f"Could not find a script for stage {stage_name}.")


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
        >>> get_parallelization_args({"n_workers": 2, "launcher_params": 'foo'}, {})
        Traceback (most recent call last):
            ...
        ValueError: If launcher_params is provided, launcher must also be provided.
        >>> get_parallelization_args({"n_workers": 2}, {})
        ['--multirun', 'worker="range(0,2)"']
        >>> get_parallelization_args(
        ...     {"launcher": "slurm"},
        ...     {"n_workers": 3, "launcher": "joblib"}
        ... )
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
    cfg: DictConfig,
    stage_name: str,
    default_parallelization_cfg: dict | DictConfig | None = None,
    runner_fn: callable = subprocess.run,  # For dependency injection
):
    """Runs a single stage of the pipeline.

    Args:
        cfg: The configuration for the entire pipeline.
        stage_name: The name of the stage to run.

    Raises:
        ValueError: If the stage fails to run.

    Examples:
        >>> def fake_shell_succeed(cmd, shell, capture_output):
        ...     print(cmd)
        ...     return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
        >>> def fake_shell_fail(cmd, shell, capture_output):
        ...     print(cmd)
        ...     return subprocess.CompletedProcess(args=cmd, returncode=1, stdout=b"", stderr=b"")
        >>> cfg = OmegaConf.create({
        ...     "pipeline_config_fp": "pipeline_config.yaml",
        ...     "do_profile": False,
        ...     "_local_pipeline_config": {
        ...         "stage_configs": {
        ...             "shard_events": {},
        ...             "fit_vocabulary_indices": {"_script": "foobar"},
        ...         },
        ...     },
        ...     "_stage_runners": {
        ...         "shard_events": {"_script": "not used"},
        ...         "fit_vocabulary_indices": {},
        ...         "baz": {"script": "baz_script"},
        ...     },
        ... })
        >>> run_stage(cfg, "shard_events", runner_fn=fake_shell_succeed) # doctest: +NORMALIZE_WHITESPACE
        MEDS_extract-shard_events --config-dir=... --config-name=pipeline_config
            'hydra.searchpath=[pkg://MEDS_transforms.configs]' stage=shard_events
        >>> run_stage(
        ...     cfg, "fit_vocabulary_indices", runner_fn=fake_shell_succeed
        ... ) # doctest: +NORMALIZE_WHITESPACE
        foobar --config-dir=... --config-name=pipeline_config
            'hydra.searchpath=[pkg://MEDS_transforms.configs]' stage=fit_vocabulary_indices
        >>> run_stage(cfg, "baz", runner_fn=fake_shell_succeed) # doctest: +NORMALIZE_WHITESPACE
        baz_script --config-dir=... --config-name=pipeline_config
            'hydra.searchpath=[pkg://MEDS_transforms.configs]' stage=baz
        >>> cfg.do_profile = True
        >>> run_stage(cfg, "baz", runner_fn=fake_shell_succeed) # doctest: +NORMALIZE_WHITESPACE
        baz_script --config-dir=... --config-name=pipeline_config
            'hydra.searchpath=[pkg://MEDS_transforms.configs]' stage=baz
            ++hydra.callbacks.profiler._target_=hydra_profiler.profiler.ProfilerCallback
        >>> cfg._stage_runners.baz.parallelize = {"n_workers": 2}
        >>> cfg.do_profile = False
        >>> run_stage(cfg, "baz", runner_fn=fake_shell_succeed) # doctest: +NORMALIZE_WHITESPACE
        baz_script --config-dir=... --config-name=pipeline_config --multirun
            'hydra.searchpath=[pkg://MEDS_transforms.configs]' stage=baz worker="range(0,2)"
        >>> run_stage(cfg, "baz", runner_fn=fake_shell_fail)
        Traceback (most recent call last):
            ...
        ValueError: Stage baz failed via ...
    """

    if default_parallelization_cfg is None:
        default_parallelization_cfg = {}

    do_profile = cfg.get("do_profile", False)
    pipeline_config_fp = Path(cfg.pipeline_config_fp)
    stage_config = cfg._local_pipeline_config.get("stage_configs", {}).get(stage_name, {})
    stage_runner_config = cfg._stage_runners.get(stage_name, {})

    script = None
    if "script" in stage_runner_config:
        script = stage_runner_config.script
    elif "_script" in stage_config:
        script = stage_config._script
    else:
        script = get_script_from_name(stage_name)

    command_parts = [
        script,
        f"--config-dir={str(pipeline_config_fp.parent.resolve())}",
        f"--config-name={pipeline_config_fp.stem}",
        "'hydra.searchpath=[pkg://MEDS_transforms.configs]'",
        f"stage={stage_name}",
    ]

    parallelization_args = get_parallelization_args(
        stage_runner_config.get("parallelize", {}), default_parallelization_cfg
    )

    if parallelization_args:
        multirun = parallelization_args.pop(0)
        command_parts = command_parts[:3] + [multirun] + command_parts[3:] + parallelization_args

    if do_profile:
        command_parts.append("++hydra.callbacks.profiler._target_=hydra_profiler.profiler.ProfilerCallback")

    full_cmd = " ".join(command_parts)
    logger.info(f"Running command: {full_cmd}")
    command_out = runner_fn(full_cmd, shell=True, capture_output=True)

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

    pipeline_config_fp = Path(cfg.pipeline_config_fp)
    if pipeline_config_fp.stem in RESERVED_CONFIG_NAMES:
        raise ValueError(
            f"Pipeline configuration file {pipeline_config_fp} must not have a name in "
            f"{RESERVED_CONFIG_NAMES}."
        )

    pipeline_config = load_yaml_file(cfg.pipeline_config_fp)
    stages = pipeline_config.get("stages", [])
    if not stages:
        raise ValueError("Pipeline configuration must specify at least one stage.")

    hydra_loguru_init()

    log_dir = Path(cfg.log_dir)

    if cfg.get("do_profile", False):  # pragma: no cover
        try:
            import hydra_profiler  # noqa: F401
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

    if "parallelize" in cfg._stage_runners:
        default_parallelization_cfg = cfg._stage_runners.parallelize
    elif "parallelize" in cfg:
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
        >>> import tempfile
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


OmegaConf.register_new_resolver("load_yaml_file", load_yaml_file, replace=True)

if __name__ == "__main__":  # pragma: no cover
    main()
