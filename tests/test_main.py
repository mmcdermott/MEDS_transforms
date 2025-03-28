import subprocess


def test_stage_entry_point_help():
    result = subprocess.run("MEDS_transform-stage", check=False, shell=True, capture_output=True)
    assert result.returncode != 0

    help_str = result.stdout.decode()
    assert "Usage: " in help_str and "Available stages:" in help_str

    result = subprocess.run("MEDS_transform-stage --help", check=False, shell=True, capture_output=True)
    assert result.returncode == 0
    assert result.stdout.decode() == help_str


def test_stage_entry_point_stage_invalid():
    result = subprocess.run(
        "MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml invalid_stage",
        check=False,
        shell=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Stage 'invalid_stage' not found." in result.stderr.decode()


def test_stage_entry_point_pipeline_invalid():
    result = subprocess.run(
        "MEDS_transform-stage not_real.yaml occlude_outliers", check=False, shell=True, capture_output=True
    )
    assert result.returncode != 0
    assert "Pipeline YAML file 'not_real.yaml' does not exist." in result.stderr.decode()
