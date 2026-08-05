import re
import subprocess
import sys
from unittest.mock import patch

SCRIPT_TEMPLATE = "MEDS_transform-stage {pipeline} {stage_name}"


def test_print_help_stage(capsys):
    from MEDS_transforms.__main__ import print_help_stage

    with patch(
        "MEDS_transforms.__main__.get_all_registered_stages",
        return_value={"bar_stage": None, "foo_stage": None},
    ):
        print_help_stage()

    captured = capsys.readouterr()
    assert "Available stages:" in captured.out
    assert "  - bar_stage" in captured.out
    assert "  - foo_stage" in captured.out


def test_stage_entry_point_help():
    result = subprocess.run("MEDS_transform-stage", check=False, shell=True, capture_output=True)
    assert result.returncode != 0

    help_str = result.stdout.decode()
    assert "Usage: " in help_str and "Available stages:" in help_str

    result = subprocess.run("MEDS_transform-stage --help", check=False, shell=True, capture_output=True)
    assert result.returncode == 0
    assert result.stdout.decode() == help_str

    result = subprocess.run("MEDS_transform-stage foo", check=False, shell=True, capture_output=True)
    assert result.returncode != 0
    assert result.stdout.decode() == help_str


def test_stage_module_is_runnable_via_dash_m():
    """`python -m MEDS_transforms` must dispatch to `run_stage`, not exit 0 having done nothing.

    Downstream ETLs invoke these entry points as `[sys.executable, "-m", ...]` so the subprocess is
    pinned to the caller's interpreter instead of being resolved off `PATH`. See issue #398.
    """

    result = subprocess.run([sys.executable, "-m", "MEDS_transforms"], check=False, capture_output=True)
    assert result.returncode == 1, "Bare `-m` invocation must print help and fail, not silently exit 0."

    help_str = result.stdout.decode()
    assert "Available stages:" in help_str
    assert "Usage: python -m MEDS_transforms <pipeline_yaml> <stage_name> [args]" in help_str

    result = subprocess.run(
        [sys.executable, "-m", "MEDS_transforms", "--help"], check=False, capture_output=True
    )
    assert result.returncode == 0
    assert result.stdout.decode() == help_str


def test_runner_module_is_runnable_via_dash_m():
    """`python -m MEDS_transforms.runner` must dispatch to `main`, not exit 0 having done nothing."""

    result = subprocess.run(
        [sys.executable, "-m", "MEDS_transforms.runner"], check=False, capture_output=True
    )
    # argparse exits 2 when the required positional is missing; the pre-fix behavior was a silent 0.
    assert result.returncode == 2, "Bare `-m` invocation must error on the missing config, not exit 0."
    assert "usage: python -m MEDS_transforms.runner" in result.stderr.decode()

    result = subprocess.run(
        [sys.executable, "-m", "MEDS_transforms.runner", "--help"], check=False, capture_output=True
    )
    assert result.returncode == 0
    assert "MEDS-Transforms Pipeline Runner" in result.stdout.decode()


def test_stage_entry_point_errors():
    for pipeline, stage, want_err in [
        ("not_real.yaml", "occlude_outliers", "Pipeline YAML file 'not_real.yaml' does not exist."),
        ("__null__", "not_real_stage", "Stage 'not_real_stage' not registered"),
        (
            "pkg://non_existent_pkg.file.yaml",
            "occlude_outliers",
            re.compile("Package 'non_existent_pkg' not found"),
        ),
    ]:
        script = SCRIPT_TEMPLATE.format(pipeline=pipeline, stage_name=stage)
        result = subprocess.run(script, check=False, shell=True, capture_output=True)
        assert result.returncode != 0
        if isinstance(want_err, str):
            assert want_err in result.stderr.decode()
        else:
            assert re.search(want_err, result.stderr.decode()) is not None
