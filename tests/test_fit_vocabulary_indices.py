"""Tests the fit vocabulary indices script.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""


from .transform_tester_base import (
    FIT_VOCABULARY_INDICES_SCRIPT,
    parse_code_metadata_csv,
    single_stage_transform_tester,
)

WANT_CSV = """
code,code/n_occurrences,code/n_subjects,values/n_occurrences,values/sum,values/sum_sqd,description,parent_codes,code/vocab_index
,44,4,28,3198.8389005974336,382968.28937288234,,,1
ADMISSION//CARDIAC,2,2,0,,,,,2
ADMISSION//ORTHOPEDIC,1,1,0,,,,,3
ADMISSION//PULMONARY,1,1,0,,,,,4
DISCHARGE,4,4,0,,,,,5
DOB,4,4,0,,,,,6
EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",,7
EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",,8
EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",,9
HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,,10
HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4,11
TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5,12
"""


def test_fit_vocabulary_indices_with_default_stage_config():
    single_stage_transform_tester(
        transform_script=FIT_VOCABULARY_INDICES_SCRIPT,
        stage_name="fit_vocabulary_indices",
        transform_stage_kwargs=None,
        want_metadata=parse_code_metadata_csv(WANT_CSV),
    )
