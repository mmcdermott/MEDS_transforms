import re
import subprocess
import sys
from unittest.mock import patch

import pytest

# The two ways the stage dispatcher can be launched. `-m` pins the subprocess to the calling
# interpreter's environment; the console script is resolved off `PATH`. See issue #398.
CONSOLE_SCRIPT = ["MEDS_transform-stage"]
DASH_M = [sys.executable, "-m", "MEDS_transforms"]

INVOCATIONS = pytest.mark.parametrize(
    "invocation", [CONSOLE_SCRIPT, DASH_M], ids=["console_script", "dash_m"]
)


def run_dispatcher(invocation: list[str], *args: str) -> subprocess.CompletedProcess:
    """Launch the stage dispatcher, without a shell so `sys.executable` is honored verbatim."""

    return subprocess.run([*invocation, *args], check=False, capture_output=True)


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


@INVOCATIONS
def test_stage_entry_point_help(invocation):
    result = run_dispatcher(invocation)
    assert result.returncode != 0

    help_str = result.stdout.decode()
    assert "Usage: " in help_str and "Available stages:" in help_str

    result = run_dispatcher(invocation, "--help")
    assert result.returncode == 0
    assert result.stdout.decode() == help_str

    result = run_dispatcher(invocation, "foo")
    assert result.returncode != 0
    assert result.stdout.decode() == help_str


@INVOCATIONS
def test_stage_entry_point_errors(invocation):
    for pipeline, stage, want_err in [
        ("not_real.yaml", "occlude_outliers", "Pipeline YAML file 'not_real.yaml' does not exist."),
        ("__null__", "not_real_stage", "Stage 'not_real_stage' not registered"),
        (
            "pkg://non_existent_pkg.file.yaml",
            "occlude_outliers",
            re.compile("Package 'non_existent_pkg' not found"),
        ),
    ]:
        result = run_dispatcher(invocation, pipeline, stage)
        assert result.returncode != 0
        if isinstance(want_err, str):
            assert want_err in result.stderr.decode()
        else:
            assert re.search(want_err, result.stderr.decode()) is not None


def test_dash_m_matches_the_console_script():
    """Both entry points must behave identically apart from naming themselves in the usage line.

    Before the `__main__` guards, `python -m MEDS_transforms` exited 0 having done nothing at all, so
    "equivalent" is precisely the property that was missing.
    """

    script = run_dispatcher(CONSOLE_SCRIPT)
    module = run_dispatcher(DASH_M)

    assert script.returncode == module.returncode == 1

    script_lines = script.stdout.decode().splitlines()
    module_lines = module.stdout.decode().splitlines()

    assert script_lines[0] == "Usage: MEDS_transform-stage <pipeline_yaml> <stage_name> [args]"
    assert module_lines[0] == "Usage: python -m MEDS_transforms <pipeline_yaml> <stage_name> [args]"
    assert script_lines[1:] == module_lines[1:]


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

    # Everything past the usage block must match the console script. The usage block itself cannot: it
    # names the invocation, and argparse re-wraps it around the differing program name.
    script_help = run_dispatcher(["MEDS_transform-pipeline"], "--help").stdout.decode()
    _, _, script_body = script_help.partition("\n\n")
    _, _, module_body = result.stdout.decode().partition("\n\n")

    assert module_body == script_body
    assert module_body.startswith("MEDS-Transforms Pipeline Runner")
