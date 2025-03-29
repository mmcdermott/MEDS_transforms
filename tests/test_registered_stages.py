import tempfile
from importlib.resources import files
from pathlib import Path

import polars as pl
import pytest
from meds import code_metadata_filepath
from meds_testing_helpers.dataset import MEDSDataset
from omegaconf import OmegaConf

from MEDS_transforms.__main__ import get_all_stages

from .utils import MEDS_transforms_pipeline_tester

# Get all registered stages
REGISTERED_STAGES = get_all_stages()
CMD_PATTERN = "MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml {stage_name}"
DO_USE_YAML_KEY = "__do_use_yaml"


def test_registered_stages(simple_static_MEDS: Path, stage: str):
    """Tests that all registered stages are present in the MEDS transforms pipeline."""

    ep = REGISTERED_STAGES[stage]

    ep_package = ep.dist.metadata["Name"]

    examples_dir = files(ep_package).joinpath("static_data_examples") / stage

    stage_cfg_fp = examples_dir / "cfg.yaml"
    in_fp = examples_dir / "in.yaml"
    want_fp = examples_dir / "out.yaml"

    if not want_fp.is_file():
        pytest.skip(f"Skipping {stage} test because example output not provided.")

    if in_fp.is_file():
        in_data = MEDSDataset.from_yaml(in_fp)
        tempdir = tempfile.TemporaryDirectory()
        input_dir = Path(tempdir.name)
        in_data.write(input_dir)

        # Currently, the MEDS testing helper only writes out columns that are in the code metadata schema. If
        # there are more, we need to write them out manually.

        in_data._pl_code_metadata.write_parquet(input_dir / code_metadata_filepath)

        # Same for data
        for k, v in in_data._pl_shards.items():
            fp = input_dir / "data" / f"{k}.parquet"
            fp.parent.mkdir(parents=True, exist_ok=True)
            v.write_parquet(fp)
    else:
        input_dir = simple_static_MEDS

    if stage == "normalization":
        want_data = MEDSDataset.from_yaml(want_fp, code=pl.Int64, numeric_value=pl.Float64)
    else:
        want_data = MEDSDataset.from_yaml(want_fp)

    if stage_cfg_fp.is_file():
        stage_kwargs = OmegaConf.to_container(OmegaConf.load(stage_cfg_fp))
    else:
        stage_kwargs = {}

    do_use_yaml = stage_kwargs.pop(DO_USE_YAML_KEY, False)

    try:
        MEDS_transforms_pipeline_tester(
            script=CMD_PATTERN.format(stage_name=stage),
            stage_name=stage,
            stage_kwargs=stage_kwargs,
            want_outputs={f"data/{k}": v for k, v in want_data._pl_shards.items()},
            input_dir=input_dir,
            do_use_config_yaml=do_use_yaml,
        )
    finally:
        if input_dir != simple_static_MEDS:
            tempdir.cleanup()
