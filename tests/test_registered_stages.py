import subprocess

from MEDS_transforms.stages import StageExample


def test_stage_help(stage: str):
    """Test the help command for a stage."""

    script = f"MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml {stage} --help"

    subprocess.run(script, shell=True, check=True)


def test_stage_scenario(stage_example: StageExample):
    stage_example.test()
