from pathlib import Path

import polars as pl

from MEDS_transforms.stages import StageExample

from .utils import MEDS_transforms_pipeline_tester

# Get all registered stages
CMD_PATTERN = "MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml {stage_name}"


def test_stage_scenario(
    stage: str,
    stage_scenario: str,
    stage_example: StageExample,
    stage_example_IO: tuple[Path, dict[str, pl.DataFrame]],
):
    input_dir, want_outputs = stage_example_IO

    MEDS_transforms_pipeline_tester(
        script=CMD_PATTERN.format(stage_name=stage),
        want_outputs=want_outputs,
        input_dir=input_dir,
        test_name=f"{stage}/{stage_scenario}",
        stages=[stage],
        stage_configs={stage: stage_example.stage_cfg},
        **stage_example.test_kwargs,
    )
