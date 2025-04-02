import importlib
import tempfile
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path

import polars as pl
import pytest
from meds import code_metadata_filepath

from . import __package_name__
from .stages import StageExample, get_all_registered_stages, get_nested_test_cases

# Get all registered stages
REGISTERED_STAGES = get_all_registered_stages()


def get_examples_for_stage(stage_name: str) -> dict[str, StageExample]:
    stage = REGISTERED_STAGES[stage_name].load()

    if not stage.examples_dir:
        return {}

    return get_nested_test_cases(stage.examples_dir, stage_name, **stage.output_schema_updates)


def pytest_addoption(parser):  # pragma: no cover
    stgaes_for_package_help_str = (
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
        "--test_stages_for_package", action="append", type=str, default=[], help=stgaes_for_package_help_str
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


def get_stages_under_test(config: pytest.Config) -> dict[str, EntryPoint]:
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
    return out


def pytest_generate_tests(metafunc):
    """Generate tests for registered stages based on the command line options."""
    allowed_stages = get_stages_under_test(metafunc.config)

    stage_scenarios = {stage: get_examples_for_stage(stage) for stage in allowed_stages.keys()}

    if "stage_scenario" in metafunc.fixturenames:
        arg_names = ["stage", "stage_scenario"]
        arg_values = [(s, sc) for s in allowed_stages.keys() for sc in stage_scenarios[s].keys()]
        metafunc.parametrize(arg_names, arg_values, scope="session")

    elif "stage" in metafunc.fixturenames:
        metafunc.parametrize("stage", list(allowed_stages.keys()), scope="session")


@pytest.fixture(scope="session")
def stage_example(stage: str, stage_scenario: str) -> StageExample:
    """Fixture to provide the example for the given stage and scenario."""

    stage_scenarios = get_examples_for_stage(stage)
    if stage_scenario not in stage_scenarios:  # pragma: no cover
        raise ValueError(f"Stage scenario '{stage_scenario}' not found for stage '{stage}'.")

    yield stage_scenarios[stage_scenario]


@pytest.fixture(scope="session")
def stage_example_on_disk(stage_example: StageExample, simple_static_MEDS: Path) -> Path:
    if stage_example.in_data is None:
        yield simple_static_MEDS
        return

    with tempfile.TemporaryDirectory() as tempdir:
        input_dir = Path(tempdir)
        stage_example.in_data.write(input_dir)

        # Currently, the MEDS testing helper only writes out columns that are in the code metadata schema.
        # If there are more, we need to write them out manually.

        stage_example.in_data._pl_code_metadata.write_parquet(input_dir / code_metadata_filepath)

        # Same for data
        for k, v in stage_example.in_data._pl_shards.items():
            fp = input_dir / "data" / f"{k}.parquet"
            fp.parent.mkdir(parents=True, exist_ok=True)
            v.write_parquet(fp)

        yield input_dir


@pytest.fixture(scope="session")
def stage_example_IO(
    stage_example: StageExample, stage_example_on_disk: Path
) -> tuple[Path, dict[str, pl.DataFrame]]:
    """Fixture to provide the input and expected output for the given stage and scenario."""
    if stage_example.want_data is not None:
        want_outputs = {f"data/{k}": v for k, v in stage_example.want_data._pl_shards.items()}
    else:
        want_outputs = {code_metadata_filepath: stage_example.want_metadata}

    yield stage_example_on_disk, want_outputs
