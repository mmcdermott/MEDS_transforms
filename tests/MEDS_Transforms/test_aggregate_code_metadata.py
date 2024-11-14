"""Tests the aggregate_code_metadata.py script.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

import polars as pl

from tests.MEDS_Transforms import AGGREGATE_CODE_METADATA_SCRIPT
from tests.MEDS_Transforms.transform_tester_base import (
    MEDS_CODE_METADATA_SCHEMA,
    MEDS_SHARDS,
    single_stage_transform_tester,
)

WANT_OUTPUT_CODE_METADATA_FILE = """
code,code/n_occurrences,code/n_subjects,values/n_occurrences,values/n_subjects,values/sum,values/sum_sqd,values/n_ints,values/min,values/max,description,parent_codes
,44,4,28,4,3198.8389005974336,382968.28937288234,6,86.0,175.271118,,
ADMISSION//CARDIAC,2,2,0,0,0,0,0,,,,
ADMISSION//ORTHOPEDIC,1,1,0,0,0,0,0,,,,
ADMISSION//PULMONARY,1,1,0,0,0,0,0,,,,
DISCHARGE,4,4,0,0,0,0,0,,,,
DOB,4,4,0,0,0,0,0,,,,
EYE_COLOR//BLUE,1,1,0,0,0,0,0,,,"Blue Eyes. Less common than brown.",
EYE_COLOR//BROWN,1,1,0,0,0,0,0,,,"Brown Eyes. The most common eye color.",
EYE_COLOR//HAZEL,2,2,0,0,0,0,0,,,"Hazel eyes. These are uncommon",
HEIGHT,4,4,4,4,656.8389005974336,108056.12937288235,0,156.485596,175.271118,,
HR,12,4,12,4,1360.5000000000002,158538.77,2,86.0,170.199997,"Heart Rate",LOINC/8867-4
TEMP,12,4,12,4,1181.4999999999998,116373.38999999998,4,95.5,100.400002,"Body Temperature",LOINC/8310-5
"""

WANT_OUTPUT_CODE_METADATA_FILE = pl.DataFrame(
    {
        "code": [
            None,
            "ADMISSION//CARDIAC",
            "ADMISSION//ORTHOPEDIC",
            "ADMISSION//PULMONARY",
            "DISCHARGE",
            "DOB",
            "EYE_COLOR//BLUE",
            "EYE_COLOR//BROWN",
            "EYE_COLOR//HAZEL",
            "HEIGHT",
            "HR",
            "TEMP",
        ],
        "code/n_occurrences": [44, 2, 1, 1, 4, 4, 1, 1, 2, 4, 12, 12],
        "code/n_subjects": [4, 2, 1, 1, 4, 4, 1, 1, 2, 4, 4, 4],
        "values/n_occurrences": [28, 0, 0, 0, 0, 0, 0, 0, 0, 4, 12, 12],
        "values/n_subjects": [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4],
        "values/sum": [
            3198.8389005974336,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            656.8389005974336,
            1360.5000000000002,
            1181.4999999999998,
        ],
        "values/sum_sqd": [
            382968.28937288234,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            108056.12937288235,
            158538.77,
            116373.38999999998,
        ],
        "values/n_ints": [6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4],
        "values/min": [86.0, None, None, None, None, None, None, None, None, 156.485596, 86.0, 95.5],
        "values/max": [
            175.271118,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            175.271118,
            170.199997,
            100.400002,
        ],
        "values/quantiles": [
            {"values/quantile/0.25": 99.9, "values/quantile/0.5": 105.1, "values/quantile/0.75": 113.4},
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            {
                "values/quantile/0.25": 160.395311,
                "values/quantile/0.5": 164.686884,
                "values/quantile/0.75": 164.686884,
            },
            {"values/quantile/0.25": 107.5, "values/quantile/0.5": 112.5, "values/quantile/0.75": 113.4},
            {"values/quantile/0.25": 96.2, "values/quantile/0.5": 99.9, "values/quantile/0.75": 100.0},
        ],
        "description": [
            None,
            None,
            None,
            None,
            None,
            None,
            "Blue Eyes. Less common than brown.",
            "Brown Eyes. The most common eye color.",
            "Hazel eyes. These are uncommon",
            None,
            "Heart Rate",
            "Body Temperature",
        ],
        "parent_codes": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ["LOINC/8867-4"],
            ["LOINC/8310-5"],
        ],
    },
    schema={
        **{k: v for k, v in MEDS_CODE_METADATA_SCHEMA.items() if k != "code/vocab_index"},
        "parent_codes": pl.List(pl.String),
        "values/quantiles": pl.Struct(
            {
                "values/quantile/0.25": pl.Float32,
                "values/quantile/0.5": pl.Float32,
                "values/quantile/0.75": pl.Float32,
            }
        ),
    },
)

MEDS_CODE_METADATA_FILE = """
code,description,parent_codes
EYE_COLOR//BLUE,"Blue Eyes. Less common than brown.",
EYE_COLOR//BROWN,"Brown Eyes. The most common eye color.",
EYE_COLOR//HAZEL,"Hazel eyes. These are uncommon",
HR,"Heart Rate",LOINC/8867-4
TEMP,"Body Temperature",LOINC/8310-5
"""

AGGREGATIONS = [
    "code/n_occurrences",
    "code/n_subjects",
    "values/n_occurrences",
    "values/n_subjects",
    "values/sum",
    "values/sum_sqd",
    "values/n_ints",
    "values/min",
    "values/max",
    {"name": "values/quantiles", "quantiles": [0.25, 0.5, 0.75]},
]


def test_aggregate_code_metadata():
    single_stage_transform_tester(
        transform_script=AGGREGATE_CODE_METADATA_SCRIPT,
        stage_name="aggregate_code_metadata",
        transform_stage_kwargs={"aggregations": AGGREGATIONS, "do_summarize_over_all_codes": True},
        want_metadata=WANT_OUTPUT_CODE_METADATA_FILE,
        input_code_metadata=MEDS_CODE_METADATA_FILE,
        do_use_config_yaml=True,
        assert_no_other_outputs=False,
        df_check_kwargs={"check_column_order": False},
    )

    # Test with shards re-mapped so it has to use the splits file.
    remapped_shards = {str(i): v for i, v in enumerate(MEDS_SHARDS.values())}
    single_stage_transform_tester(
        transform_script=AGGREGATE_CODE_METADATA_SCRIPT,
        stage_name="aggregate_code_metadata",
        transform_stage_kwargs={"aggregations": AGGREGATIONS, "do_summarize_over_all_codes": True},
        want_metadata=WANT_OUTPUT_CODE_METADATA_FILE,
        input_code_metadata=MEDS_CODE_METADATA_FILE,
        do_use_config_yaml=True,
        assert_no_other_outputs=False,
        df_check_kwargs={"check_column_order": False},
        input_shards=remapped_shards,
    )

    single_stage_transform_tester(
        transform_script=AGGREGATE_CODE_METADATA_SCRIPT,
        stage_name="aggregate_code_metadata",
        transform_stage_kwargs={"aggregations": AGGREGATIONS, "do_summarize_over_all_codes": True},
        want_metadata=WANT_OUTPUT_CODE_METADATA_FILE,
        input_code_metadata=MEDS_CODE_METADATA_FILE,
        do_use_config_yaml=True,
        input_shards=remapped_shards,
        splits_fp=None,
        should_error=True,
    )
