"""This script is a helper utility to run entire pipelines from a single script.

To do this effectively, this runner functionally takes a "meta configuration" file that contains:

  1. The path to the pipeline configuration file.
  2. Configuration details for how to run each stage of the pipeline, including mappings to the underlying
     stage scripts and Hydra launcher configurations for each stage to control parallelism, resources, etc.
"""

import argparse
import logging
import shlex
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from . import __package_name__
from .configs import PipelineConfig
from .configs.utils import OmegaConfResolver
from .in_memory import FrameStore, in_memory_mode
from .stage_launch import run_stage_in_process
from .stage_outputs import carry_forward
from .utils import invocation_name

# Spelled out rather than taken from ``__name__``, which is ``"__main__"`` in exactly the case being
# named. Unlike the package's ``__main__.py``, whose ``-m`` target is the package itself, this module is
# its own ``-m`` target.
_MODULE_NAME = f"{__package_name__}.runner"

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

logger = logging.getLogger(__name__)


def stage_dir_overrides(stage_name: str, prior_stage_name: str | None) -> list[str]:
    """The overrides that point a stage at its own directories within the pipeline's output directory.

    Run on its own, a stage reads ``input_dir`` and writes ``output_dir``. In a pipeline, each stage instead
    writes the MEDS dataset ``${output_dir}/<stage>`` and reads the one written by the stage before it (the
    first stage reads ``input_dir``, as it would on its own). Its logs go to ``${output_dir}/.logs/<stage>``,
    next to the runner's own, so they are kept if the stage's directory is removed. Once every stage has
    finished, the runner publishes the last stage's dataset as ``${output_dir}/data`` and
    ``${output_dir}/metadata`` (see `publish_final_outputs`).

    The paths are left as interpolations of ``output_dir``, so they follow any override of it. Quote each
    override (e.g. with ``shlex.quote``) before putting it in a shell command. These are plain overrides of
    keys every stage's config defines (see ``configs/_main.yaml``), so a custom stage ``script`` whose
    config lacks them fails at launch rather than ignoring them and writing somewhere the next stage won't
    look.

    Args:
        stage_name: The stage to run.
        prior_stage_name: The stage before it in the pipeline, or ``None`` if it is the first.

    Returns:
        The overrides, unquoted.

    Examples:
        >>> for o in stage_dir_overrides("count_codes", None):
        ...     print(o)
        stage_output_dir=${output_dir}/count_codes
        log_dir=${output_dir}/.logs/count_codes
        >>> for o in stage_dir_overrides("filter_measurements", "count_codes"):
        ...     print(o)
        stage_input_dir=${output_dir}/count_codes
        stage_output_dir=${output_dir}/filter_measurements
        log_dir=${output_dir}/.logs/filter_measurements
    """
    overrides = []
    if prior_stage_name is not None:
        overrides.append(f"stage_input_dir=${{output_dir}}/{prior_stage_name}")
    overrides.append(f"stage_output_dir=${{output_dir}}/{stage_name}")
    overrides.append(f"log_dir=${{output_dir}}/.logs/{stage_name}")
    return overrides


def run_stage_in_memory(
    store: FrameStore,
    pipeline_config_fp: str,
    stage_name: str,
    prior_in_memory_stage: str | None,
    overrides: list[str],
    output_dir: Path,
) -> None:
    """Run a stage in process with its data output held in ``store``.

    The stage's ``data/`` output directory becomes one of the store's roots before it runs. Afterwards, any
    shards the stage wrote directly to disk there (through its own write function) are taken over by the
    store, and the previous stage's in-memory data is dropped, as nothing reads it any more. No ``.done``
    marker is written: the stage's data exists only in this process, so a later run could not resume
    from it.

    Args:
        store: The active in-memory store.
        pipeline_config_fp: The pipeline YAML.
        stage_name: The stage to run.
        prior_in_memory_stage: The previous stage, if it too ran in memory in this run.
        overrides: The stage's Hydra overrides, unquoted.
        output_dir: The pipeline's output directory.

    Examples:
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
        ...     overrides = ["stage=filter_subjects", *stage_dir_overrides("filter_subjects", None)]
        ...     with in_memory_mode() as store:
        ...         run_stage_in_memory(store, pipeline_fp, "filter_subjects", None, overrides, output_dir)
        ...         data_dir = output_dir / "filter_subjects" / "data"
        ...         print(sorted(p.as_posix() for p in store.files(data_dir)))
        ...         print(data_dir.exists(), (output_dir / "filter_subjects" / "metadata").is_dir())
        ['held_out/0.parquet', 'train/0.parquet', 'train/1.parquet', 'tuning/0.parquet']
        False True
    """
    data_dir = output_dir / stage_name / "data"
    store.add_root(data_dir)
    before = store.disk_snapshot(data_dir)
    run_stage_in_process(pipeline_config_fp, stage_name, overrides)
    if adopted := store.adopt_disk_files(data_dir, before):
        logger.info(f"Stage {stage_name} wrote {len(adopted)} shard(s) to disk; reading them from there.")
    if prior_in_memory_stage is not None:
        store.remove_root(output_dir / prior_in_memory_stage / "data")


def check_in_process_compatible(
    pipeline_cfg: PipelineConfig, stage_runner_fp: str | None, do_profile: bool
) -> None:
    """Raise if the pipeline asks for something only stage subprocesses can do.

    In-process execution runs each stage once, as a single worker, by calling it directly, so it can't
    launch a custom stage ``script``, apply a Hydra launcher's parallelization, or attach Hydra's profiler
    callback. Rather than silently ignore such settings, it refuses them.

    Args:
        pipeline_cfg: The pipeline to run.
        stage_runner_fp: The ``--stage_runner_fp`` argument, if given.
        do_profile: The ``--do_profile`` argument.

    Raises:
        ValueError: If any of those settings is present.

    Examples:
        >>> check_in_process_compatible(PipelineConfig(stages=["filter_subjects"]), None, False)
        >>> check_in_process_compatible(PipelineConfig(stages=["filter_subjects"]), "runner.yaml", False)
        Traceback (most recent call last):
            ...
        ValueError: --in_process does not support --stage_runner_fp: stage runners configure subprocesses.
        >>> check_in_process_compatible(PipelineConfig(stages=["filter_subjects"]), None, True)
        Traceback (most recent call last):
            ...
        ValueError: --in_process does not support --do_profile: profiling is a Hydra launch callback.
        >>> parallel = PipelineConfig(stages=["filter_subjects"], additional_params={"parallelize": {}})
        >>> check_in_process_compatible(parallel, None, False)
        Traceback (most recent call last):
            ...
        ValueError: --in_process does not support the pipeline's `parallelize` setting: it runs one worker.
        >>> scripted = PipelineConfig(stages=[{"filter_subjects": {"_script": "my_script"}}])
        >>> check_in_process_compatible(scripted, None, False)
        Traceback (most recent call last):
            ...
        ValueError: --in_process can't run stage filter_subjects, which sets a custom `_script`.
    """
    if stage_runner_fp is not None:
        raise ValueError(
            "--in_process does not support --stage_runner_fp: stage runners configure subprocesses."
        )
    if do_profile:
        raise ValueError("--in_process does not support --do_profile: profiling is a Hydra launch callback.")
    if "parallelize" in (pipeline_cfg.additional_params or {}):
        raise ValueError(
            "--in_process does not support the pipeline's `parallelize` setting: it runs one worker."
        )
    for stage in pipeline_cfg.parsed_stages:
        if "_script" in stage.config:
            raise ValueError(f"--in_process can't run stage {stage.name}, which sets a custom `_script`.")


def publish_final_outputs(output_dir: Path, last_stage_name: str) -> None:
    """Publish the last stage's dataset as the pipeline's result, ``output_dir/{data,metadata}``.

    The result is made to mirror ``output_dir/<last stage>`` exactly, with hard links where possible (see
    `MEDS_transforms.stage_outputs.carry_forward`), so it costs no extra disk space and holds no files left
    over from an earlier run of a different pipeline into the same directory.

    Args:
        output_dir: The pipeline's output directory.
        last_stage_name: The name of the pipeline's last stage.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     out = Path(tmpdir)
        ...     _ = yaml_disk(
        ...         {
        ...             "count_codes/": {
        ...                 "data/train/": {"0.txt": "shard"},
        ...                 "metadata/": {"codes.txt": "codes"},
        ...                 ".scratch/": {"partial.txt": "partial"},
        ...             },
        ...             "data/": {"stale.txt": "from an earlier run"},
        ...         },
        ...         out,
        ...     )
        ...     publish_final_outputs(out, "count_codes")
        ...     print_directory(out)
        ...     print((out / "data" / "train" / "0.txt").samefile(out / "count_codes/data/train/0.txt"))
        ├── count_codes
        │   ├── .scratch
        │   │   └── partial.txt
        │   ├── data
        │   │   └── train
        │   │       └── 0.txt
        │   └── metadata
        │       └── codes.txt
        ├── data
        │   └── train
        │       └── 0.txt
        └── metadata
            └── codes.txt
        True
    """
    for half in ("data", "metadata"):
        src, dst = output_dir / last_stage_name / half, output_dir / half
        n = len(carry_forward(src, dst, mirror=True))
        logger.info(f"Published {src} as {dst} ({n} file(s) updated)")


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
    cfg_overrides: list[str] | None = None,
    default_parallelization_cfg: dict | DictConfig | None = None,
    do_profile: bool = False,
    runner_fn: callable = subprocess.run,  # For dependency injection
):
    """Runs a single stage of the pipeline.

    Args:
        pipeline_config_fp: Path to the pipeline configuration on disk.
        stage_runners_cfg: The dictionary of stage runner configurations.
        stage_name: The name of the stage to run.
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

        Paths with spaces are properly quoted:

        >>> run_stage(
        ...     "/path/to/my pipeline/config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reshard_to_split",
        ...     runner_fn=fake_shell_succeed,
        ... )
        MEDS_transform-stage '/path/to/my pipeline/config.yaml' reshard_to_split stage=reshard_to_split
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
        script = f"MEDS_transform-stage {shlex.quote(str(pipeline_config_fp))} {stage_name}"

    command_parts = [
        script,
        f"stage={stage_name}",
    ]

    if cfg_overrides:
        command_parts.extend(cfg_overrides)

    parallelization_args = get_parallelization_args(
        stage_runner_config.get("parallelize", {}), default_parallelization_cfg
    )

    if parallelization_args:
        multirun = parallelization_args.pop(0)
        command_parts = [*command_parts[:1], multirun, *command_parts[1:], *parallelization_args]

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
            f"Stage {stage_name} failed via {full_cmd} with return code {command_out.returncode}.\n"
            f"Stdout:\n{stdout}\nStderr:\n{stderr}"
        )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Run an entire pipeline based on command line arguments."""

    from ._cli_help import pipeline_keys_help_block

    parser = argparse.ArgumentParser(
        prog=invocation_name(_MODULE_NAME),
        description="MEDS-Transforms Pipeline Runner",
        epilog=pipeline_keys_help_block(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pipeline_config_fp",
        help="Path to the pipeline configuration file, either as a raw path or with pkg:// syntax.",
    )
    parser.add_argument(
        "--stage_runner_fp",
        type=str,
        default=None,
        help="Path to the stage runner configuration file. If not provided, no specialized runner is used.",
    )
    parser.add_argument(
        "--do_profile",
        default=False,
        # No `type=`: `BooleanOptionalAction` already yields a bool, and pairing the two is deprecated
        # as of 3.12 and removed in 3.14. Running `main` in-process (rather than only via subprocess)
        # is what surfaced the warning.
        action=argparse.BooleanOptionalAction,
        help="Enable Hydra profiling for the stages.",
    )
    parser.add_argument(
        "--in_process",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Run every stage in this process, as a single worker, rather than launching a "
            "`MEDS_transform-stage` subprocess per stage. Outputs are the same; stage runner configs, "
            "parallelization and profiling are not supported."
        ),
    )
    parser.add_argument(
        "--in_memory",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Like --in_process, but keep each stage's data shards in memory rather than on disk. Only the "
            "final dataset and the logs are written; a failed run restarts from the beginning."
        ),
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[], help="Additional overrides for the pipeline configuration."
    )
    args = parser.parse_args(argv)
    if args.in_memory:
        args.in_process = True

    pipeline_config = PipelineConfig.from_arg(args.pipeline_config_fp, args.overrides)
    if pipeline_config.additional_params is None or "output_dir" not in pipeline_config.additional_params:
        raise ValueError("Pipeline configuration or override must specify an 'output_dir'")

    log_dir = Path(pipeline_config.additional_params["output_dir"]) / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=(log_dir / "pipeline.log"), level=logging.INFO)

    logging.info("Running MEDS-Transforms Pipeline Runner with the following arguments:")
    for arg_name, arg_value in vars(args).items():
        logging.info(f"  {arg_name}: {arg_value}")

    global_done_file = log_dir / "_all_stages.done"
    if global_done_file.exists():
        logger.info("All stages are already complete. Exiting.")
        return 0

    stages = [s.name for s in pipeline_config.parsed_stages]
    if not stages:
        raise ValueError("Pipeline configuration must specify at least one stage.")

    if args.in_process:
        check_in_process_compatible(pipeline_config, args.stage_runner_fp, args.do_profile)

    stage_runners_cfg = load_yaml_file(args.stage_runner_fp)

    if "parallelize" in stage_runners_cfg:
        logger.info("Parallelization configuration loaded from stage runner")
        default_parallelization_cfg = stage_runners_cfg["parallelize"]
    elif "parallelize" in pipeline_config.additional_params:
        logger.info("Parallelization configuration loaded from pipeline config")
        default_parallelization_cfg = pipeline_config.additional_params["parallelize"]
    else:
        logger.info("No parallelization configuration provided.")
        default_parallelization_cfg = None

    output_dir = log_dir.parent
    in_memory_stages = []
    with in_memory_mode() if args.in_memory else nullcontext() as store:
        for prior_stage, stage in zip([None, *stages[:-1]], stages, strict=True):
            done_file = log_dir / f"{stage}.done"

            if done_file.exists():
                logger.info(f"Skipping stage {stage} as it is already complete.")
                continue

            logger.info(f"Running stage: {stage}")
            dir_overrides = stage_dir_overrides(stage, prior_stage)
            if store is not None:
                run_stage_in_memory(
                    store,
                    args.pipeline_config_fp,
                    stage,
                    prior_stage if prior_stage in in_memory_stages else None,
                    [f"stage={stage}", *dir_overrides, *args.overrides],
                    output_dir,
                )
                in_memory_stages.append(stage)
                continue

            if args.in_process:
                run_stage_in_process(
                    args.pipeline_config_fp, stage, [f"stage={stage}", *dir_overrides, *args.overrides]
                )
            else:
                run_stage(
                    args.pipeline_config_fp,
                    stage_runners_cfg,
                    pipeline_config,
                    stage,
                    cfg_overrides=[*(shlex.quote(o) for o in dir_overrides), *args.overrides],
                    default_parallelization_cfg=default_parallelization_cfg,
                    do_profile=args.do_profile,
                )
            done_file.touch()

        # Inside the in-memory block, so the last stage's in-memory shards are written out to disk.
        publish_final_outputs(output_dir, stages[-1])

    for stage in in_memory_stages:
        # These directories hold only the stage's metadata and scratch files; its data was never on disk,
        # and the published result links to (or copies) the last stage's metadata.
        shutil.rmtree(output_dir / stage, ignore_errors=True)
    global_done_file.touch()

    if pipeline_config.additional_params.get("cleanup_intermediates", False):
        # Each stage writes its output under `<output_dir>/<stage>`, and the pipeline's result was published
        # to `<output_dir>/{data,metadata}` as links to (or copies of) the last stage's files, so every stage
        # directory can go. A removed stage's `.done` marker goes too, so a re-run recomputes it.
        for stage in stages:
            if (stage_dir := log_dir.parent / stage).is_dir():
                logger.info(f"Removing intermediate stage output {stage_dir}")
                shutil.rmtree(stage_dir)
                (log_dir / f"{stage}.done").unlink(missing_ok=True)
    return 0


# `replace=True` so re-executing this module is idempotent. It is executed a second time whenever it is
# entered as `__main__` (`python -m MEDS_transforms.runner`) in a process that has already imported it;
# without this, registering the resolver again raises.
@OmegaConfResolver(replace=True)
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


if __name__ == "__main__":
    sys.exit(main())
