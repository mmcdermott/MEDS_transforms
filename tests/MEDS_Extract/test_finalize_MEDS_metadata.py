"""Tests the finalize MEDS metadata process.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from datetime import datetime

import polars as pl
from meds import __version__ as MEDS_VERSION
from meds import code_metadata_filepath, dataset_metadata_filepath, subject_splits_filepath

from MEDS_transforms.utils import get_package_version as get_meds_transform_version
from tests.MEDS_Extract import FINALIZE_METADATA_SCRIPT
from tests.utils import MEDS_transforms_pipeline_tester

SHARDS_JSON = {
    "train/0": [239684, 1195293],
    "train/1": [68729, 814703],
    "tuning/0": [754281],
    "held_out/0": [1500733],
}

WANT_OUTPUTS = {
    "metadata/codes": pl.DataFrame(
        {
            "code": ["EYE_COLOR//BLUE", "EYE_COLOR//BROWN", "EYE_COLOR//HAZEL", "HR", "TEMP"],
            "description": [
                "Blue Eyes. Less common than brown.",
                "Brown Eyes. The most common eye color.",
                "Hazel eyes. These are uncommon",
                "Heart Rate",
                "Body Temperature",
            ],
            "parent_codes": [None, None, None, ["LOINC/8867-4"], ["LOINC/8310-5"]],
        }
    ),
}

METADATA_DF = pl.DataFrame(
    {
        "code": ["EYE_COLOR//BLUE", "EYE_COLOR//BROWN", "EYE_COLOR//HAZEL", "HR", "TEMP"],
        "description": [
            "Blue Eyes. Less common than brown.",
            "Brown Eyes. The most common eye color.",
            "Hazel eyes. These are uncommon",
            "Heart Rate",
            "Body Temperature",
        ],
        "parent_codes": [None, None, None, ["LOINC/8867-4"], ["LOINC/8310-5"]],
    }
)


def want_dataset_metadata(got: dict):
    want_known = {
        "dataset_name": "TEST",
        "dataset_version": "1.0",
        "etl_name": "MEDS_transforms",
        "etl_version": get_meds_transform_version(),
        "meds_version": MEDS_VERSION,
    }

    assert "created_at" in got, "Expected 'created_at' to be in the dataset metadata."
    created_at_obs = got.pop("created_at")
    as_dt = datetime.fromisoformat(created_at_obs)
    assert as_dt < datetime.now(), f"Expected 'created_at' to be before now, got {created_at_obs}."
    created_ago = datetime.now() - as_dt
    assert created_ago.total_seconds() < 5 * 60, "Expected 'created_at' to be within 5 minutes of now."

    assert got == want_known, f"Expected dataset metadata (less created at) to be {want_known}, got {got}."


WANT_OUTPUTS = {
    code_metadata_filepath: (
        METADATA_DF.with_columns(
            pl.col("code").cast(pl.String),
            pl.col("description").cast(pl.String),
            pl.col("parent_codes").cast(pl.List(pl.String)),
        ).select(["code", "description", "parent_codes"])
    ),
    subject_splits_filepath: pl.DataFrame(
        {
            "subject_id": [239684, 1195293, 68729, 814703, 754281, 1500733],
            "split": ["train", "train", "train", "train", "tuning", "held_out"],
        }
    ),
    dataset_metadata_filepath: want_dataset_metadata,
}


def test_convert_to_sharded_events():
    MEDS_transforms_pipeline_tester(
        script=FINALIZE_METADATA_SCRIPT,
        stage_name="finalize_MEDS_metadata",
        stage_kwargs=None,
        config_name="extract",
        input_files={
            "metadata/codes": METADATA_DF,
            "metadata/.shards.json": SHARDS_JSON,
        },
        **{"etl_metadata.dataset_name": "TEST", "etl_metadata.dataset_version": "1.0"},
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        shards_map_fp="{input_dir}/metadata/.shards.json",
        want_outputs=WANT_OUTPUTS,
        df_check_kwargs={"check_row_order": False, "check_column_order": True, "check_dtypes": True},
    )
