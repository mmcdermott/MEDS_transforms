import tempfile
from importlib.resources import files
from pathlib import Path

import polars as pl
import pytest
from meds import code_metadata_filepath

from MEDS_transforms.stages import StageExample, get_all_stages, get_nested_test_cases

from .utils import MEDS_transforms_pipeline_tester

try:
    pass
except ImportError:  # pragma: no cover
    pass

# Get all registered stages
REGISTERED_STAGES = get_all_stages()
CMD_PATTERN = "MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml {stage_name}"


def test_registered_stages(simple_static_MEDS: Path, stage: str):
    """Tests that all registered stages are present in the MEDS transforms pipeline."""

    ep = REGISTERED_STAGES[stage]

    ep_package = ep.dist.metadata["Name"]

    examples_dir = files(ep_package).joinpath("stages") / stage

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
                ),
                "code/n_occurrences": pl.UInt8,
                "code/n_subjects": pl.UInt8,
                "values/n_occurrences": pl.UInt8,
                "values/n_subjects": pl.UInt8,
                "values/sum": pl.Float32,
                "values/sum_sqd": pl.Float32,
                "values/n_ints": pl.UInt8,
                "values/min": pl.Float32,
                "values/max": pl.Float32,
            }
        case "fit_vocabulary_indices":
            schema_updates = {"code/vocab_index": pl.UInt8}
        case _:
            schema_updates = {}

    examples: dict[str, StageExample] = get_nested_test_cases(examples_dir, stage, **schema_updates)

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
                want_outputs=want_outputs,
                input_dir=input_dir,
                test_name=name,
                stages=[stage],
                stage_configs={stage: example.stage_cfg},
                **example.test_kwargs,
            )
        finally:
            if input_dir != simple_static_MEDS:
                tempdir.cleanup()
