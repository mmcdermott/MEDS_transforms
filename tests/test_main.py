import doctest
import re
import subprocess
from unittest.mock import patch

SCRIPT_TEMPLATE = "MEDS_transform-stage {pipeline} {stage_name}"


def test_dispatcher_doctests():
    """Run the doctests in `MEDS_transforms.__main__`.

    `--doctest-modules` deliberately skips any file named `__main__.py` (see `_pytest.doctest`), so the
    examples in the dispatcher would otherwise never execute and would be free to rot.
    """

    import MEDS_transforms.__main__ as dispatcher

    results = doctest.testmod(
        dispatcher, optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS, verbose=False
    )
    assert results.attempted > 0, "No doctests were found in the dispatcher module."
    assert results.failed == 0


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
