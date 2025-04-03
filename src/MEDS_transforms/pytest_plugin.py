import importlib
import tomllib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from . import __package_name__
from .stages import StageExample, get_all_registered_stages

# Get all registered stages
REGISTERED_STAGES = get_all_registered_stages()


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
    detected_package = _auto_detect_package(config)
    config.detected_package = detected_package

    cli_packages = config.getoption("test_stages_for_package")

    # Set packages_to_test based on CLI or detection
    if cli_packages:
        packages_to_test = cli_packages
    elif detected_package:
        packages_to_test = [detected_package]
    else:
        packages_to_test = None  # will handle later

    config.packages_to_test = packages_to_test


def _auto_detect_package(config: pytest.Config) -> str | None:
    """Attempts to automatically detect the package name based on the pytest Config's rootpath."""
    rootpath = Path(config.rootpath)
    pyproject_file = rootpath / "pyproject.toml"
    setup_file = rootpath / "setup.py"

    if pyproject_file.exists():
        try:
            with open(pyproject_file, "rb") as f:
                pyproject = tomllib.load(f)
                return pyproject.get("project", {}).get("name")
        except Exception:  # pragma: no cover
            pass
    if setup_file.exists():
        spec = importlib.util.spec_from_file_location("setup", setup_file)
        setup_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(setup_module)
            return getattr(setup_module, "__name__", None)
        except Exception:  # pragma: no cover
            pass
    return None


def get_stages_under_test(config: pytest.Config) -> dict[str, dict[str, StageExample]]:
    packages = config.packages_to_test

    if packages is None:
        out = {n: ep for n, ep in REGISTERED_STAGES.items() if ep.dist.metadata["Name"] != __package_name__}
    else:
        out = {n: ep for n, ep in REGISTERED_STAGES.items() if ep.dist.metadata["Name"] in packages}

    stages = config.getoption("test_stage")
    if stages:
        if any(stage not in out for stage in stages):
            raise ValueError(
                f"Invalid stage(s) specified: {', '.join(s for s in stages if s not in out)}. "
                f"Available stages given specified package ({packages}) are: {', '.join(out.keys())}."
            )
        out = {stage: ep for stage, ep in out.items() if stage in stages}

    out = {n: ep.load().test_cases for n, ep in out.items()}
    return out


def pytest_generate_tests(metafunc):
    """Generate tests for registered stages based on the command line options."""
    config = metafunc.config

    config.allowed_stage_scenarios = get_stages_under_test(config)

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

    yield request.config.allowed_stage_scenarios[stage][stage_scenario]


import subprocess
import tempfile


def pipeline_tester(pipeline_yaml: str, stage_runner_yaml: str, stage_scenario_sequence: list[str]):
    """Test the pipeline with the given YAML configuration and stage scenario sequence."""

    pipeline_stages = list(OmegaConf.create(pipeline_yaml).stages)

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

        stage_runner_fp = test_root / "stage_runner.yaml"
        stage_runner_fp.write_text(stage_runner_yaml)

        command = [
            "MEDS_transform-pipeline",
            f"pipeline_config_fp={pipeline_config_fp}",
            f"stage_runner_fp={stage_runner_fp}",
        ]

        # 2. Run the pipeline
        out = subprocess.run(
            command,
            shell=False,
            check=False,
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
