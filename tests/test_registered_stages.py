import tempfile
from dataclasses import dataclass
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


@dataclass
class StageExample:
    out_data: MEDSDataset
    stage_cfg: dict
    in_data: MEDSDataset | None = None
    do_use_yaml: bool = False


NAMED_EXAMPLES = dict[str, StageExample]


def parse_example_dir(example_dir: Path, **schema_updates) -> StageExample:
    """Parse the example directory and return a StageExample object, or raise an error if invalid."""

    stage_cfg_fp = example_dir / "cfg.yaml"
    in_fp = example_dir / "in.yaml"
    want_fp = example_dir / "out.yaml"

    if not want_fp.is_file():
        raise FileNotFoundError(f"Output file not found: {want_fp}")

    out_data = MEDSDataset.from_yaml(want_fp, **schema_updates)
    in_data = MEDSDataset.from_yaml(in_fp) if in_fp.is_file() else None
    stage_cfg = OmegaConf.to_container(OmegaConf.load(stage_cfg_fp)) if stage_cfg_fp.is_file() else {}
    do_use_yaml = stage_cfg.pop(DO_USE_YAML_KEY, False)

    return StageExample(
        out_data=out_data,
        stage_cfg=stage_cfg,
        in_data=in_data,
        do_use_yaml=do_use_yaml,
    )


def is_example_dir(path: Path) -> bool:
    try:
        parse_example_dir(path)
        return True
    except FileNotFoundError:
        return False


def get_nested_test_cases(
    example_dir: Path,
    prefix: str | None = None,
    test_cases: NAMED_EXAMPLES | None = None,
    **schema_updates,
) -> NAMED_EXAMPLES:
    """Recursively get all test cases from the example directory."""
    if test_cases is None:
        test_cases = {}
    if prefix is None:
        prefix = ""

    if not example_dir.is_dir():
        return test_cases

    if is_example_dir(example_dir):
        test_cases[prefix] = parse_example_dir(example_dir, **schema_updates)
        return test_cases

    for path in example_dir.iterdir():
        if path.is_dir():
            nested_prefix = f"{prefix}/{path.name}"
            get_nested_test_cases(path, nested_prefix, test_cases, **schema_updates)

    return test_cases


def test_registered_stages(simple_static_MEDS: Path, stage: str):
    """Tests that all registered stages are present in the MEDS transforms pipeline."""

    ep = REGISTERED_STAGES[stage]

    ep_package = ep.dist.metadata["Name"]

    examples_dir = files(ep_package).joinpath("static_data_examples") / stage

    schema_updates = {} if stage != "normalization" else {"code": pl.Int64, "numeric_value": pl.Float64}

    examples = get_nested_test_cases(examples_dir, stage, **schema_updates)

    if not examples:
        pytest.skip(f"Skipping {stage} test because no examples provided.")

    for name, example in examples.items():
        if example.in_data is not None:
            tempdir = tempfile.TemporaryDirectory()
            input_dir = Path(tempdir.name)
            example.in_data.write(input_dir)

            # Currently, the MEDS testing helper only writes out columns that are in the code metadata schema.
            # If there are more, we need to write them out manually.

            example.in_data._pl_code_metadata.write_parquet(input_dir / code_metadata_filepath)

            # Same for data
            for k, v in example.in_data._pl_shards.items():
                fp = input_dir / "data" / f"{k}.parquet"
                fp.parent.mkdir(parents=True, exist_ok=True)
                v.write_parquet(fp)
        else:
            input_dir = simple_static_MEDS

        want_data = example.out_data

        try:
            MEDS_transforms_pipeline_tester(
                script=CMD_PATTERN.format(stage_name=stage),
                stage_name=stage,
                stage_kwargs=example.stage_cfg,
                want_outputs={f"data/{k}": v for k, v in want_data._pl_shards.items()},
                input_dir=input_dir,
                do_use_config_yaml=example.do_use_yaml,
                test_name=name,
            )
        finally:
            if input_dir != simple_static_MEDS:
                tempdir.cleanup()
