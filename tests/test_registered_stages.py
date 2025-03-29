import copy
import tempfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from yaml import load as load_yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader

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
    stage_cfg: dict
    want_data: MEDSDataset | None = None
    want_metadata: pl.DataFrame | None = None
    in_data: MEDSDataset | None = None
    do_use_yaml: bool = False

    def __post_init__(self):
        if self.want_data is None and self.want_metadata is None:
            raise ValueError("Either want_data or want_metadata must be provided.")


NAMED_EXAMPLES = dict[str, StageExample]


def read_metadata_only(fp: Path, **schema_updates) -> pl.DataFrame:
    data = load_yaml(fp.read_text(), Loader=Loader)
    assert len(data) == 1
    key = list(data.keys())[0]
    assert key == code_metadata_filepath

    return MEDSDataset.parse_csv(data[key], **copy.deepcopy(schema_updates))


def parse_example_dir(example_dir: Path, **schema_updates) -> StageExample:
    """Parse the example directory and return a StageExample object, or raise an error if invalid."""

    stage_cfg_fp = example_dir / "cfg.yaml"
    in_fp = example_dir / "in.yaml"
    want_fp = example_dir / "out.yaml"

    if not want_fp.is_file():
        raise FileNotFoundError(f"Output file not found: {want_fp}")

    try:
        want_data = MEDSDataset.from_yaml(want_fp, **copy.deepcopy(schema_updates))
        want_metadata = None
    except ValueError as e:
        if "No data shards found in YAML" not in str(e):
            raise e
        want_data = None
        try:
            want_metadata = read_metadata_only(want_fp, **copy.deepcopy(schema_updates))
        except Exception as e2:
            raise e2 from e

    in_data = MEDSDataset.from_yaml(in_fp) if in_fp.is_file() else None
    stage_cfg = OmegaConf.to_container(OmegaConf.load(stage_cfg_fp)) if stage_cfg_fp.is_file() else {}
    do_use_yaml = stage_cfg.pop(DO_USE_YAML_KEY, False)

    return StageExample(
        want_data=want_data,
        want_metadata=want_metadata,
        stage_cfg=stage_cfg,
        in_data=in_data,
        do_use_yaml=do_use_yaml,
    )


def is_example_dir(path: Path) -> bool:
    want_fp = path / "out.yaml"
    return want_fp.is_file()


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
        test_cases[prefix] = parse_example_dir(example_dir, **copy.deepcopy(schema_updates))
        return test_cases

    for path in example_dir.iterdir():
        if path.is_dir():
            nested_prefix = f"{prefix}/{path.name}"
            get_nested_test_cases(path, nested_prefix, test_cases, **copy.deepcopy(schema_updates))

    return test_cases


def test_registered_stages(simple_static_MEDS: Path, stage: str):
    """Tests that all registered stages are present in the MEDS transforms pipeline."""

    ep = REGISTERED_STAGES[stage]

    ep_package = ep.dist.metadata["Name"]

    examples_dir = files(ep_package).joinpath("static_data_examples") / stage

    match stage:
        case "normalization":
            schema_updates = {"code": pl.Int64, "numeric_value": pl.Float64}
        case "aggregate_code_metadata":
            schema_updates = {
                "values/quantiles": pl.Struct(
                    {
                        "values/quantile/0.25": pl.Float32,
                        "values/quantile/0.5": pl.Float32,
                        "values/quantile/0.75": pl.Float32,
                    }
                )
            }
        case _:
            schema_updates = {}

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

        if example.want_data is not None:
            want_outputs = {f"data/{k}": v for k, v in example.want_data._pl_shards.items()}
        else:
            want_outputs = {code_metadata_filepath: example.want_metadata}

        try:
            MEDS_transforms_pipeline_tester(
                script=CMD_PATTERN.format(stage_name=stage),
                stage_name=stage,
                stage_kwargs=example.stage_cfg,
                want_outputs=want_outputs,
                input_dir=input_dir,
                do_use_config_yaml=example.do_use_yaml,
                test_name=name,
            )
        finally:
            if input_dir != simple_static_MEDS:
                tempdir.cleanup()
