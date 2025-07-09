"""This file will use the automated stage example definitions to test each defined stage in this package."""

from MEDS_transforms.stages import StageExample


def test_stage_scenario(stage_example: StageExample):
    stage_example.test()
