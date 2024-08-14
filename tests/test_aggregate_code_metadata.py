"""Tests the aggregate_code_metadata.py script.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from .transform_tester_base import (
    AGGREGATE_CODE_METADATA_SCRIPT,
    parse_code_metadata_csv,
    single_stage_transform_tester,
)

WANT_OUTPUT_CODE_METADATA_FILE = """
code,code/n_occurrences,code/n_patients,values/n_occurrences,values/sum,values/sum_sqd,description,parent_codes
,44,4,28,3198.8389005974336,382968.28937288234,,
ADMISSION//CARDIAC,2,2,0,0,0,,
ADMISSION//ORTHOPEDIC,1,1,0,0,0,,
ADMISSION//PULMONARY,1,1,0,0,0,,
DISCHARGE,4,4,0,0,0,,
DOB,4,4,0,0,0,,
EYE_COLOR//BLUE,1,1,0,0,0,"Blue Eyes. Less common than brown.",
EYE_COLOR//BROWN,1,1,0,0,0,"Brown Eyes. The most common eye color.",
EYE_COLOR//HAZEL,2,2,0,0,0,"Hazel eyes. These are uncommon",
HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
"""

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
    "code/n_patients",
    "values/n_occurrences",
    "values/sum",
    "values/sum_sqd",
]


def test_aggregate_code_metadata():
    single_stage_transform_tester(
        transform_script=AGGREGATE_CODE_METADATA_SCRIPT,
        stage_name="aggregate_code_metadata",
        transform_stage_kwargs={"aggregations": AGGREGATIONS, "do_summarize_over_all_codes": True},
        want_outputs=parse_code_metadata_csv(WANT_OUTPUT_CODE_METADATA_FILE),
        code_metadata=MEDS_CODE_METADATA_FILE,
        do_use_config_yaml=True,
    )
