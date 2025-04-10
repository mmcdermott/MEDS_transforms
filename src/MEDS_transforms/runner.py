"""This script is a helper utility to run entire pipelines from a single script.

To do this effectively, this runner functionally takes a "meta configuration" file that contains:

  1. The path to the pipeline configuration file.
  2. Configuration details for how to run each stage of the pipeline, including mappings to the underlying
     stage scripts and Hydra launcher configurations for each stage to control parallelism, resources, etc.
"""

import logging
from pathlib import Path
import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml

from .configs import RUNNER_CONFIG_YAML, PipelineConfig
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
    cfg: DictConfig,
    pipeline_cfg: PipelineConfig,
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
        >>> cfg = DictConfig(
        ...     {
        ...         "pipeline_config_fp": "pipeline_config.yaml",
        ...         "do_profile": False,
        ...         "_stage_runners": {
        ...             "reshard_to_split": {"_script": "not used"},
        ...             "fit_vocabulary_indices": {},
        ...             "reorder_measurements": {"script": "baz_script"},
        ...         },
        ...     }
        ... )
        >>> pipeline_cfg = PipelineConfig(
        ...     stages=[
        ...         "reshard_to_split",
        ...         {"fit_vocabulary_indices": {"_script": "foobar"}},
        ...         "reorder_measurements",
        ...     ],
        ... )
        >>> run_stage(cfg, pipeline_cfg, "reshard_to_split", runner_fn=fake_shell_succeed)
        MEDS_transform-stage pipeline_config.yaml reshard_to_split stage=reshard_to_split
        >>> run_stage(cfg, pipeline_cfg, "fit_vocabulary_indices", runner_fn=fake_shell_succeed)
        foobar stage=fit_vocabulary_indices
        >>> run_stage(cfg, pipeline_cfg, "reorder_measurements", runner_fn=fake_shell_succeed)
        baz_script stage=reorder_measurements
        >>> cfg.do_profile = True
        >>> run_stage(cfg, pipeline_cfg, "reorder_measurements", runner_fn=fake_shell_succeed)
        baz_script stage=reorder_measurements
            ++hydra.callbacks.profiler._target_=hydra_profiler.profiler.ProfilerCallback
        >>> cfg._stage_runners.reorder_measurements.parallelize = {"n_workers": 2}
        >>> cfg.do_profile = False
        >>> run_stage(cfg, pipeline_cfg, "reorder_measurements", runner_fn=fake_shell_succeed)
        baz_script --multirun stage=reorder_measurements worker="range(0,2)"
        >>> run_stage(cfg, pipeline_cfg, "reorder_measurements", runner_fn=fake_shell_fail)
        Traceback (most recent call last):
            ...
        ValueError: Stage reorder_measurements failed via ...

        Errors are also raised if the stage runner is improperly configured.

        >>> cfg = DictConfig(
        ...     {
        ...         "pipeline_config_fp": "pipeline_config.yaml",
        ...         "do_profile": False,
        ...         "_stage_runners": {
        ...             "reshard_to_split": {"_base_stage": "belongs in the stage"},
        ...         },
        ...     }
        ... )
        >>> pipeline_cfg = PipelineConfig(stages=["reshard_to_split"])
        >>> run_stage(cfg, pipeline_cfg, "reshard_to_split", runner_fn=fake_shell_succeed)
        Traceback (most recent call last):
            ...
        ValueError: Put _base_stage args is in your pipeline config
    """

    if default_parallelization_cfg is None:
        default_parallelization_cfg = {}

    do_profile = cfg.get("do_profile", False)
    stage_config = pipeline_cfg.parsed_stages_by_name[stage_name].config
    stage_runner_config = cfg._stage_runners.get(stage_name, {})

    script = None
    if "script" in stage_runner_config:
        script = stage_runner_config.get("script")
    elif "_script" in stage_config:
        script = stage_config.get("_script")
    elif "_base_stage" in stage_runner_config:
        raise ValueError("Put _base_stage args is in your pipeline config")
    else:
        script = f"MEDS_transform-stage {cfg.pipeline_config_fp} {stage_name}"

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


@hydra.main(
    version_base=None, config_path=str(RUNNER_CONFIG_YAML.parent), config_name=RUNNER_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """Runs the entire pipeline, end-to-end, based on the configuration provided.

    This script will launch many subsidiary commands via `subprocess`, one for each stage of the specified
    pipeline.

    Args:
        cfg: The Hydra configuration for the entire pipeline. See `configs/_runner.yaml` for details.

    Raises:
        ValueError: If the pipeline configuration is invalid, if any stage fails to run, or if you try to run
            with `do_profile` without the `hydra-profiler` package installed.

    Examples:
        >>> cfg = DictConfig(
        ...     {
        ...         "pipeline_config_fp": "???",  # we'll fill this in each test.
        ...         "log_dir": "???",  # We'll fill this in each test; it is added by Hydra during CLI usage.
        ...         "do_profile": False,
        ...         "_stage_runners": {},
        ...     }
        ... )
        >>> pipeline_cfg = DictConfig(
        ...     {
        ...         "stages": [
        ...             "reshard_to_split",
        ...             {"count_codes": {"_base_stage": "aggregate_code_metadata"}},
        ...         ],
        ...     }
        ... )

    We'll consistently mock out the `run_stage` calls to avoid running the actual stages; instead, when
    called, we'll just print out the stages. To do this, we'll set up a helper to run the full test.

        >>> def mock_run_stage(cfg, pipeline_cfg, stage, *args, **kwargs):
        ...     if "default_parallelization_cfg" in kwargs and kwargs["default_parallelization_cfg"]:
        ...         print(f"Running {stage} with: {kwargs['default_parallelization_cfg']}")
        ...     else:
        ...         print(f"Running {stage}")
        >>> def profile_main(test_dir: Path):
        ...     # Copy the cfg
        ...     test_cfg = DictConfig(cfg)
        ...     # Make th log dir
        ...     log_dir = test_dir / "logs"
        ...     log_dir.mkdir(exist_ok=True, parents=True)
        ...     test_cfg.log_dir = str(log_dir)
        ...     # Save the pipeline config
        ...     pipeline_cfg_fp = test_dir / "pipeline_config.yaml"
        ...     OmegaConf.save(pipeline_cfg, pipeline_cfg_fp)
        ...     test_cfg.pipeline_config_fp = str(pipeline_cfg_fp)
        ...     # Mock the run_stage function
        ...     with patch("MEDS_transforms.runner.run_stage") as mock:
        ...         mock.side_effect = mock_run_stage
        ...         # Run the main function
        ...         main(test_cfg)

    Now we can run some tests. Let's start with the runner in normal operation. We see it "runs" each stage
    (here just prints them):

        >>> with tempfile.TemporaryDirectory() as test_dir:
        ...     test_dir = Path(test_dir)
        ...     profile_main(test_dir)
        Running reshard_to_split
        Running count_codes

    What controls exist? Firstly, upon completion of each stage, it also writes a ".done" file to the log dir
    for that stage, and a global "_all_stages.done" file when the pipeline is done. Let's see them:

        >>> with tempfile.TemporaryDirectory() as test_dir:
        ...     test_dir = Path(test_dir)
        ...     profile_main(test_dir)
        ...     print("Log contents:")
        ...     print_directory_contents(test_dir / "logs")
        Running reshard_to_split
        Running count_codes
        Log contents:
        ├── _all_stages.done
        ├── count_codes.done
        └── reshard_to_split.done

    If the global done file exists in advance, the pipeline will skip all stages:

        >>> with tempfile.TemporaryDirectory() as test_dir:
        ...     test_dir = Path(test_dir)
        ...     log_dir = test_dir / "logs"
        ...     log_dir.mkdir()
        ...     global_done_file = log_dir / "_all_stages.done"
        ...     global_done_file.touch()
        ...     profile_main(test_dir)
        ...     print("Log contents:")
        ...     print_directory_contents(log_dir)
        Log contents:
        └── _all_stages.done

    This applies to each individual stage's done file too:

        >>> with tempfile.TemporaryDirectory() as test_dir:
        ...     test_dir = Path(test_dir)
        ...     log_dir = test_dir / "logs"
        ...     log_dir.mkdir()
        ...     reshard_done_file = log_dir / "reshard_to_split.done"
        ...     reshard_done_file.touch()
        ...     profile_main(test_dir)
        ...     print("Log contents:")
        ...     print_directory_contents(log_dir)
        Running count_codes
        Log contents:
        ├── _all_stages.done
        ├── count_codes.done
        └── reshard_to_split.done

    We can also control parallelization, either through the pipeline config or the stage runner config:

        >>> cfg.parallelize = {"n_workers": 2}
        >>> with tempfile.TemporaryDirectory() as test_dir:
        ...     profile_main(Path(test_dir))
        Running reshard_to_split with: {'n_workers': 2}
        Running count_codes with: {'n_workers': 2}

    Stage runner parallelization takes priority, though:

        >>> cfg._stage_runners.parallelize = {"n_workers": 3}
        >>> with tempfile.TemporaryDirectory() as test_dir:
        ...     profile_main(Path(test_dir))
        Running reshard_to_split with: {'n_workers': 3}
        Running count_codes with: {'n_workers': 3}

    The runner will throw an error if the pipeline speified has no stages:

        >>> pipeline_cfg.stages = []
        >>> with tempfile.TemporaryDirectory() as test_dir:
        ...     profile_main(Path(test_dir))
        Traceback (most recent call last):
            ...
        ValueError: Pipeline configuration must specify at least one stage.
    """

    pipeline_config = PipelineConfig.from_arg(cfg.pipeline_config_fp)

    stages = [s.name for s in pipeline_config.parsed_stages]
    if not stages:
        raise ValueError("Pipeline configuration must specify at least one stage.")

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
            run_stage(cfg, pipeline_config, stage, default_parallelization_cfg=default_parallelization_cfg)
            done_file.touch()

    global_done_file.touch()


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
