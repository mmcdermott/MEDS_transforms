"""Base helper code and data inputs for all transforms integration tests."""

from collections import defaultdict
from io import StringIO
from pathlib import Path

import polars as pl
from meds import subject_id_field, subject_splits_filepath

from tests.utils import FILE_T, MEDS_transforms_pipeline_tester, parse_shards_yaml

# Test MEDS data (inputs)

SHARDS = {
    "train/0": [239684, 1195293],
    "train/1": [68729, 814703],
    "tuning/0": [754281],
    "held_out/0": [1500733],
}

SPLITS_DF = pl.DataFrame(
    {
        subject_id_field: [239684, 1195293, 68729, 814703, 754281, 1500733],
        "split": ["train", "train", "train", "train", "tuning", "held_out"],
    }
)

MEDS_SHARDS = parse_shards_yaml(
    """
train/0: |-2
  subject_id,time,code,numeric_value
  239684,,EYE_COLOR//BROWN,
  239684,,HEIGHT,175.271115221764
  239684,"12/28/1980, 00:00:00",DOB,
  239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
  239684,"05/11/2010, 17:41:51",HR,102.6
  239684,"05/11/2010, 17:41:51",TEMP,96.0
  239684,"05/11/2010, 17:48:48",HR,105.1
  239684,"05/11/2010, 17:48:48",TEMP,96.2
  239684,"05/11/2010, 18:25:35",HR,113.4
  239684,"05/11/2010, 18:25:35",TEMP,95.8
  239684,"05/11/2010, 18:57:18",HR,112.6
  239684,"05/11/2010, 18:57:18",TEMP,95.5
  239684,"05/11/2010, 19:27:19",DISCHARGE,
  1195293,,EYE_COLOR//BLUE,
  1195293,,HEIGHT,164.6868838269085
  1195293,"06/20/1978, 00:00:00",DOB,
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
  1195293,"06/20/2010, 19:23:52",HR,109.0
  1195293,"06/20/2010, 19:23:52",TEMP,100.0
  1195293,"06/20/2010, 19:25:32",HR,114.1
  1195293,"06/20/2010, 19:25:32",TEMP,100.0
  1195293,"06/20/2010, 19:45:19",HR,119.8
  1195293,"06/20/2010, 19:45:19",TEMP,99.9
  1195293,"06/20/2010, 20:12:31",HR,112.5
  1195293,"06/20/2010, 20:12:31",TEMP,99.8
  1195293,"06/20/2010, 20:24:44",HR,107.7
  1195293,"06/20/2010, 20:24:44",TEMP,100.0
  1195293,"06/20/2010, 20:41:33",HR,107.5
  1195293,"06/20/2010, 20:41:33",TEMP,100.4
  1195293,"06/20/2010, 20:50:04",DISCHARGE,

train/1: |-2
  subject_id,time,code,numeric_value
  68729,,EYE_COLOR//HAZEL,
  68729,,HEIGHT,160.3953106166676
  68729,"03/09/1978, 00:00:00",DOB,
  68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,
  68729,"05/26/2010, 02:30:56",HR,86.0
  68729,"05/26/2010, 02:30:56",TEMP,97.8
  68729,"05/26/2010, 04:51:52",DISCHARGE,
  814703,,EYE_COLOR//HAZEL,
  814703,,HEIGHT,156.48559093209357
  814703,"03/28/1976, 00:00:00",DOB,
  814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,
  814703,"02/05/2010, 05:55:39",HR,170.2
  814703,"02/05/2010, 05:55:39",TEMP,100.1
  814703,"02/05/2010, 07:02:30",DISCHARGE,

tuning/0: |-2
  subject_id,time,code,numeric_value
  754281,,EYE_COLOR//BROWN,
  754281,,HEIGHT,166.22261567137025
  754281,"12/19/1988, 00:00:00",DOB,
  754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
  754281,"01/03/2010, 06:27:59",HR,142.0
  754281,"01/03/2010, 06:27:59",TEMP,99.8
  754281,"01/03/2010, 08:22:13",DISCHARGE,

held_out/0: |-2
  subject_id,time,code,numeric_value
  1500733,,EYE_COLOR//BROWN,
  1500733,,HEIGHT,158.60131573580904
  1500733,"07/20/1986, 00:00:00",DOB,
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
  1500733,"06/03/2010, 14:54:38",HR,91.4
  1500733,"06/03/2010, 14:54:38",TEMP,100.0
  1500733,"06/03/2010, 15:39:49",HR,84.4
  1500733,"06/03/2010, 15:39:49",TEMP,100.3
  1500733,"06/03/2010, 16:20:49",HR,90.1
  1500733,"06/03/2010, 16:20:49",TEMP,100.1
  1500733,"06/03/2010, 16:44:26",DISCHARGE,
    """
)

MEDS_CODE_METADATA_CSV = """
code,code/n_occurrences,code/n_subjects,values/n_occurrences,values/sum,values/sum_sqd,description,parent_codes
,44,4,28,3198.8389005974336,382968.28937288234,,
ADMISSION//CARDIAC,2,2,0,,,,
ADMISSION//ORTHOPEDIC,1,1,0,,,,
ADMISSION//PULMONARY,1,1,0,,,,
DISCHARGE,4,4,0,,,,
DOB,4,4,0,,,,
EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",
EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",
EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",
HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
"""

MEDS_CODE_METADATA_SCHEMA = {
    "code": pl.Utf8,
    "code/n_occurrences": pl.UInt8,
    "code/n_subjects": pl.UInt8,
    "values/n_occurrences": pl.UInt8,
    "values/n_subjects": pl.UInt8,
    "values/sum": pl.Float32,
    "values/sum_sqd": pl.Float32,
    "values/n_ints": pl.UInt8,
    "values/min": pl.Float32,
    "values/max": pl.Float32,
    "description": pl.Utf8,
    "parent_codes": pl.Utf8,
    "code/vocab_index": pl.UInt8,
}


def parse_code_metadata_csv(csv_str: str) -> pl.DataFrame:
    cols = csv_str.strip().split("\n")[0].split(",")
    schema = {col: dt for col, dt in MEDS_CODE_METADATA_SCHEMA.items() if col in cols}
    df = pl.read_csv(StringIO(csv_str), schema=schema)
    if "parent_codes" in cols:
        df = df.with_columns(pl.col("parent_codes").cast(pl.List(pl.Utf8)))
    return df


MEDS_CODE_METADATA = parse_code_metadata_csv(MEDS_CODE_METADATA_CSV)


def remap_inputs_for_transform(
    input_code_metadata: pl.DataFrame | str | None = None,
    input_shards: dict[str, pl.DataFrame] | None = None,
    input_shards_map: dict[str, list[int]] | None = None,
    input_splits_map: dict[str, list[int]] | None = None,
    splits_fp: Path | str | None = subject_splits_filepath,
) -> dict[str, FILE_T]:
    unified_inputs = {}

    if input_code_metadata is None:
        input_code_metadata = MEDS_CODE_METADATA
    elif isinstance(input_code_metadata, str):
        input_code_metadata = parse_code_metadata_csv(input_code_metadata)

    unified_inputs["metadata/codes.parquet"] = input_code_metadata

    if input_shards is None:
        input_shards = MEDS_SHARDS

    for shard_name, df in input_shards.items():
        unified_inputs[f"data/{shard_name}.parquet"] = df

    if input_shards_map is None:
        input_shards_map = SHARDS

    unified_inputs["metadata/.shards.json"] = input_shards_map

    if input_splits_map is None:
        input_splits_map = SPLITS_DF

    if isinstance(input_splits_map, pl.DataFrame):
        input_splits_df = input_splits_map
    else:
        input_splits_as_df = defaultdict(list)
        for split_name, subject_ids in input_splits_map.items():
            input_splits_as_df[subject_id_field].extend(subject_ids)
            input_splits_as_df["split"].extend([split_name] * len(subject_ids))

        input_splits_df = pl.DataFrame(input_splits_as_df)

    if splits_fp is not None:
        # This case is added for error testing; not for general use.
        unified_inputs[splits_fp] = input_splits_df

    return unified_inputs


def single_stage_transform_tester(
    transform_script: str | Path,
    stage_name: str,
    transform_stage_kwargs: dict[str, str] | None,
    do_pass_stage_name: bool = False,
    do_use_config_yaml: bool = False,
    want_data: dict[str, pl.DataFrame] | None = None,
    want_metadata: pl.DataFrame | None = None,
    assert_no_other_outputs: bool = True,
    should_error: bool = False,
    df_check_kwargs: dict | None = None,
    **input_data_kwargs,
):
    if df_check_kwargs is None:
        df_check_kwargs = {}

    base_kwargs = {
        "script": transform_script,
        "stage_name": stage_name,
        "stage_kwargs": transform_stage_kwargs,
        "do_pass_stage_name": do_pass_stage_name,
        "do_use_config_yaml": do_use_config_yaml,
        "assert_no_other_outputs": assert_no_other_outputs,
        "should_error": should_error,
        "config_name": "preprocess",
        "input_files": remap_inputs_for_transform(**input_data_kwargs),
        "df_check_kwargs": df_check_kwargs,
    }

    want_outputs = {}
    if want_data:
        for data_fn, want in want_data.items():
            want_outputs[f"data/{data_fn}"] = want
    if want_metadata is not None:
        want_outputs["metadata/codes.parquet"] = want_metadata

    base_kwargs["want_outputs"] = want_outputs

    MEDS_transforms_pipeline_tester(**base_kwargs)
