import subprocess


def test_stage_entry_point_help():
    result = subprocess.run("MEDS_transform-stage", check=False, shell=True, capture_output=True)
    assert result.returncode != 0

    help_str = result.stdout.decode()
    assert "Usage: " in help_str and "Available stages:" in help_str

    result = subprocess.run("MEDS_transform-stage --help", check=False, shell=True, capture_output=True)
    assert result.returncode == 0
    assert result.stdout.decode() == help_str


def test_stage_entry_point_invalid():
    result = subprocess.run(
        "MEDS_transform-stage invalid_stage", check=False, shell=True, capture_output=True
    )
    assert result.returncode != 0
    assert "Stage 'invalid_stage' not found." in result.stderr.decode()
