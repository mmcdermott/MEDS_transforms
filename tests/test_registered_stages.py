from importlib.resources import files
from pathlib import Path

import pytest
from meds_testing_helpers.dataset import MEDSDataset
from omegaconf import OmegaConf

from MEDS_transforms.__main__ import get_all_stages

from .utils import MEDS_transforms_pipeline_tester

# Get all registered stages
REGISTERED_STAGES = get_all_stages()
CMD_PATTERN = "MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml {stage_name}"


@pytest.mark.parametrize("stage", sorted(list(REGISTERED_STAGES.keys())))
def test_registered_stages(simple_static_MEDS: Path, stage: str):
    """Tests that all registered stages are present in the MEDS transforms pipeline."""

    ep = REGISTERED_STAGES[stage]

    ep_package = ep.dist.metadata["Name"]

    examples_dir = files(ep_package).joinpath("static_data_examples") / stage

    stage_cfg_fp = examples_dir / "cfg.yaml"
    want_fp = examples_dir / "out.yaml"

    if not want_fp.is_file():
        pytest.skip(f"Skipping {stage} test because example output not provided.")

    want_data = MEDSDataset.from_yaml(want_fp)

    stage_kwargs = OmegaConf.to_container(OmegaConf.load(stage_cfg_fp))

    MEDS_transforms_pipeline_tester(
        script=CMD_PATTERN.format(stage_name=stage),
        stage_name=stage,
        stage_kwargs=stage_kwargs,
        want_outputs={f"data/{k}": v for k, v in want_data._pl_shards.items()},
        input_dir=simple_static_MEDS,
    )
