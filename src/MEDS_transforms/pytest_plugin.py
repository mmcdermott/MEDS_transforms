import logging
import subprocess
import tempfile
import tomllib
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from . import __package_name__
from .configs.stage import StageConfig
from .stages import StageExample, get_all_registered_stages

logger = logging.getLogger(__name__)

# Get all registered stages
REGISTERED_STAGES = get_all_registered_stages()


@contextmanager
def suppress_logging(level: int = logging.CRITICAL):
    """Suppress logging at the specified level."""
    logger = logging.getLogger()
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def pytest_addoption(parser):  # pragma: no cover
    stages_for_package_help_str = (
        "Specify the package name(s) to search within when looking for registered stages to test. "
        "If not specified, all registered stages, _except for the core MEDS-Transforms stages_ will be "
        "tested. This option will attempt to automatically apply an intelligent default to just the current "
        "package being tested by pytest, but this may fail, in which case you can set this manually. "
        "This option can be used multiple times to specify multiple packages; e.g., "
        "'--test_stage_for_package package1 --test_stage_for_package package2'. "
        "Packages specified here but that lack any stages or any stages with static data examples will be "
        "skipped without error."
    )
    parser.addoption(
        "--test_stages_for_package", action="append", type=str, default=[], help=stages_for_package_help_str
    )

    stage_help_str = (
        "Specify the registered MEDS-Transforms stage(s) to test. If not specified, all registered "
        "stages will be run. This option can be used multiple times to specify multiple stages; e.g., "
        "'--test_stage stage1 --test_stage stage2'. The stages are specified by their registered names. "
        "This option only affects the automated testing of all registered stages, and does not impact other "
        "tests, such as the runner or any manual tests. Stages specified here but that lack static data "
        "examples will be skipped."
    )
    parser.addoption("--test_stage", action="append", type=str, default=[], help=stage_help_str)


def pytest_configure(config: pytest.Config):
    """Configure pytest by attaching the appropriate package to use to the config object."""

    if cli_packages := config.getoption("test_stages_for_package"):
        packages_to_test = cli_packages
    elif detected_package := _auto_detect_package(Path(config.rootpath)):
        packages_to_test = [detected_package]
    else:
        packages_to_test = []

    stages = config.getoption("test_stage")

    allowed_stages = {}
    for n, ep in REGISTERED_STAGES.items():
        pkg = ep.dist.metadata["Name"]

        no_packages_specified = len(packages_to_test) == 0
        pkg_valid = (no_packages_specified and (pkg != __package_name__)) or (pkg in packages_to_test)

        stage_valid = (not stages) or (n in stages)

        if pkg_valid and stage_valid:
            allowed_stages[n] = ep.load()

    missing_stages = sorted(list(set(stages) - set(allowed_stages.keys())))

    if missing_stages:
        raise ValueError(
            f"Invalid stage(s) specified: {', '.join(missing_stages)}. "
            f"Available stages given specified package ({packages_to_test}) are: {', '.join(allowed_stages)}."
        )

    config.allowed_stages = allowed_stages
    with suppress_logging():
        config.allowed_stage_scenarios = {n: s.test_cases for n, s in allowed_stages.items()}


def _auto_detect_package(root: Path) -> str | None:
    """Attempts to automatically detect the package name based on the pytest Config's rootpath.

    This is a best-effort attempt to find the package name by looking for a pyproject.toml file in the
    rootpath of the pytest config (passed directly). If this file is not found, or if the package name cannot
    be determined, `None` is returned.

    > [!Warning]
    > This function _only_ works with `pyproject.toml` files that are stored in the `root` path. `setup.py`
    > files are not supported!

    Args:
        root: The rootpath of the pytest session.

    Returns:
        The package name if found, otherwise None.

    Examples:

        If we find a pyproject.toml file, we can use that to get the package name.

        >>> toml_contents = '''
        ... [project]
        ... name = "my_package"
        ... '''
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     pyproject_fp = Path(tmpdir) / "pyproject.toml"
        ...     _ = pyproject_fp.write_text(toml_contents)
        ...     print(_auto_detect_package(Path(tmpdir)))
        my_package

        If we can't find a pyproject.toml file or if it can't be parsed, we return None.

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     print(_auto_detect_package(Path(tmpdir))) # No toml file!
        None
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     pyproject_fp = Path(tmpdir) / "pyproject.toml"
        ...     _ = pyproject_fp.write_text("Invalid TOML")
        ...     print(_auto_detect_package(Path(tmpdir)))
        None

        If a setup.py file is found, we warn the user that it is not supported and return None.

        >>> setup_contents = '''
        ... from setuptools import setup
        ... setup(
        ...     name="my_package",
        ...     version="0.1",
        ...     packages=["my_package"],
        ... )
        ... '''
        >>> with tempfile.TemporaryDirectory() as tmpdir, print_warnings():
        ...     setup_fp = Path(tmpdir) / "setup.py"
        ...     _ = setup_fp.write_text(setup_contents)
        ...     print(_auto_detect_package(Path(tmpdir)))
        None
        Warning: setup.py file found in /tmp/tmp..., but no pyproject.toml. Auto detection of package
        name only works with pyproject.toml files currently, so `None` will be returned.
    """

    pyproject_file = root / "pyproject.toml"
    setup_file = root / "setup.py"

    if pyproject_file.exists():
        try:
            with open(pyproject_file, "rb") as f:
                pyproject = tomllib.load(f)
                return pyproject.get("project", {}).get("name")
        except Exception:  # pragma: no cover
            pass
    elif setup_file.exists():
        logger.warning(
            f"setup.py file found in {root}, but no pyproject.toml. Auto detection of package name only "
            "works with pyproject.toml files currently, so `None` will be returned."
        )
    return None


def pytest_generate_tests(metafunc):
    """Generate tests for registered stages based on the command line options."""
    config = metafunc.config

    if "stage_scenario" in metafunc.fixturenames:
        arg_names = ["stage", "stage_scenario"]
        arg_values = [(s, sc) for s, scenarios in config.allowed_stage_scenarios.items() for sc in scenarios]
        metafunc.parametrize(arg_names, arg_values, scope="session")

    elif "stage" in metafunc.fixturenames:
        metafunc.parametrize("stage", list(config.allowed_stage_scenarios.keys()), scope="session")


@pytest.fixture(scope="session")
def stage_example(request: pytest.FixtureRequest, stage: str, stage_scenario: str) -> StageExample:
    """Fixture to provide the example for the given stage and scenario."""

    if stage_scenario not in request.config.allowed_stage_scenarios[stage]:  # pragma: no cover
        raise ValueError(f"Stage scenario '{stage_scenario}' not found for stage '{stage}'.")

    s = request.config.allowed_stages[stage]
    if s.stage_name != stage:  # pragma: no cover
        raise ValueError(
            f"Stage '{stage}' has a misconfigured registration point!. It is registered at {stage}, "
            f"but the loaded stage name is {s.name}."
        )

    yield request.config.allowed_stage_scenarios[stage][stage_scenario]


def pipeline_tester(
    pipeline_yaml: str,
    stage_runner_yaml: str | None,
    stage_scenario_sequence: list[str],
    run_fn: Callable | None = subprocess.run,
):
    """Test the pipeline with the given YAML configuration and stage scenario sequence.

    Args:
        pipeline_yaml: A string with the yaml for the pipeline configuration to be tested. If you would like
            to support a file or pkg file syntax here, file a GitHub issue.
        stage_runner_yaml: A string with the yaml for the stage runner configuration to be tested. If you
            would like to support a file or pkg file syntax here, file a GitHub issue.
        stage_scenario_sequence: A list of strings of the form `"{stage_name}/{scenario}"` where `"scenario"`
            may be "/" separated or non-existent (so just "{stage_name}") that define the input/output
            relationships expected for the pipeline for each stage -- e.g., the input to the first listed
            scenario is the test input to the pipeline, its output is expected to be the output that the
            pipeline produces when it runs the first stage, which is then fed into the second stage, etc.
            These are used to test the pipeline conforms to expectations across all stages.
        run_fn: A callable to run the pipeline. Defaults to `subprocess.run`. This is useful for dependency
            injection during testing.

    Raises:
        ValueError: If the pipeline YAML and stage scenario sequence do not match in length.
        AssertionError: If the pipeline fails to produce expected output for any stage.

    Examples:

        We just show error cases here for now, but check out the `tests/test_pipeline.py` file to see this in
        action with the default stages.

        >>> pipeline_yaml = "stages: ['foo', 'bar']"
        >>> pipeline_tester(pipeline_yaml, "", ["just_one"])
        Traceback (most recent call last):
            ...
        ValueError: Incorrect pipeline test specification! Pipeline YAML has 2 stages, but 1 stage scenarios
        were provided.
        >>> pipeline_tester(pipeline_yaml, "", ["foo/just_one", "bar/just_one"])
        Traceback (most recent call last):
            ...
        ValueError: Error loading stage example for ...

        To see it in action, we'll use fake run functions.

        >>> def fake_run_success(script, shell, capture_output):
        ...     return subprocess.CompletedProcess([], returncode=0, stdout=b"Success", stderr=b"")
        >>> def fake_run_failure(script, shell, capture_output):
        ...     return subprocess.CompletedProcess([], returncode=1, stdout=b"", stderr=b"Failure")

        We'll use the real, default example for the `filter_subjects` stage here.

        >>> pipeline_yaml = "stages: [filter_subjects]" # This wouldn't work in real life

        If we run and throw an error from the fake shell runner, it will raise an AssertionError.

        >>> pipeline_tester(pipeline_yaml, "", ["filter_subjects"], run_fn=fake_run_failure)
        Traceback (most recent call last):
            ...
        AssertionError: Pipeline failed with error: Pipeline returned code 1.
        Stdout:
        <BLANKLINE>
        Stderr:
        Failure

        If we run and succeed, it will still throw an error as the stage will fail to validate its expected
        files.

        >>> pipeline_tester(pipeline_yaml, "", ["filter_subjects"], run_fn=fake_run_success)
        Traceback (most recent call last):
            ...
        AssertionError: Pipeline failed to produce expected output for stage 'filter_subjects'
    """

    pipeline_stages = [StageConfig.from_arg(s).name for s in OmegaConf.create(pipeline_yaml).stages]

    if len(pipeline_stages) != len(stage_scenario_sequence):
        raise ValueError(
            "Incorrect pipeline test specification! "
            f"Pipeline YAML has {len(pipeline_stages)} stages, but {len(stage_scenario_sequence)} "
            "stage scenarios were provided."
        )

    stage_examples = []
    for stage_scenario in stage_scenario_sequence:
        parts = stage_scenario.split("/")
        if len(parts) > 1:
            stage_name, scenario_name = parts[0], "/".join(parts[1:])
        else:
            stage_name = stage_scenario
            scenario_name = "."

        try:
            stage_examples.append(REGISTERED_STAGES[stage_name].load().test_cases[scenario_name])
        except Exception as e:
            raise ValueError(f"Error loading stage example for {stage_name}/{scenario_name}") from e

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Set-up
        test_root = Path(tmpdir)

        input_dir = test_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        stage_examples[0].write_for_test(input_dir)

        cohort_dir = test_root / "cohort"

        pipeline_yaml = pipeline_yaml.format(input_dir=input_dir, cohort_dir=cohort_dir)

        pipeline_config_fp = test_root / "pipeline.yaml"
        pipeline_config_fp.write_text(pipeline_yaml)

        command = [
            "MEDS_transform-pipeline",
            f"pipeline_config_fp={pipeline_config_fp}",
        ]

        if stage_runner_yaml is not None:
            stage_runner_fp = test_root / "stage_runner.yaml"
            stage_runner_fp.write_text(stage_runner_yaml)
            command.append(f"stage_runner_fp={stage_runner_fp}")

        # 2. Run the pipeline
        out = run_fn(
            command,
            shell=False,
            capture_output=True,
        )

        def err_msg(m: str) -> str:
            lines = [
                f"Pipeline failed with error: {m}",
                "Stdout:",
                out.stdout.decode("utf-8"),
                "Stderr:",
                out.stderr.decode("utf-8"),
            ]
            return "\n".join(lines)

        assert out.returncode == 0, err_msg(f"Pipeline returned code {out.returncode}.")

        # 3. Check the output
        last_data_stage = (None, None)
        last_metadata_stage = (None, None)

        for name, stage in zip(pipeline_stages, stage_examples):
            if stage.want_data is not None:
                last_data_stage = (name, stage)
            if stage.want_metadata is not None:
                last_metadata_stage = (name, stage)

        for name, stage in zip(pipeline_stages, stage_examples):
            is_last_data_stage = last_data_stage[0] == name
            is_last_metadata_stage = last_metadata_stage[0] == name

            stage_output_dir = cohort_dir / name

            stage.df_check_kwargs = {"rtol": 5e-2, "atol": 1e-2, "check_dtypes": False}

            try:
                if stage.want_data is not None:
                    if is_last_data_stage:
                        stage.check_outputs(cohort_dir)
                    else:
                        stage.check_outputs(stage_output_dir, is_resolved_dir=True)

                if stage.want_metadata is not None:
                    if is_last_metadata_stage:
                        stage.check_outputs(cohort_dir)
                    else:
                        stage.check_outputs(stage_output_dir, is_resolved_dir=True)
            except AssertionError as e:
                raise AssertionError(f"Pipeline failed to produce expected output for stage '{name}'") from e
