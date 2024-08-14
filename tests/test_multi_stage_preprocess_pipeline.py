"""Tests a multi-stage pre-processing pipeline. Only checks the end result, not the intermediate files.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.

In this test, the following stages are run:
  - filter_patients
  - add_time_derived_measurements
  - fit_outlier_detection
  - occlude_outliers
  - fit_normalization
  - fit_vocabulary_indices
  - normalization
  - tokenization
  - tensorization

The stage configuration arguments will be as given in the yaml block below:
"""


from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from .transform_tester_base import (
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT,
    AGGREGATE_CODE_METADATA_SCRIPT,
    FILTER_PATIENTS_SCRIPT,
    FIT_VOCABULARY_INDICES_SCRIPT,
    NORMALIZATION_SCRIPT,
    OCCLUDE_OUTLIERS_SCRIPT,
    TENSORIZATION_SCRIPT,
    TOKENIZATION_SCRIPT,
    multi_stage_transform_tester,
    parse_shards_yaml,
)

MEDS_CODE_METADATA_FILE = """
code,description,parent_codes
EYE_COLOR//BLUE,"Blue Eyes. Less common than brown.",
EYE_COLOR//BROWN,"Brown Eyes. The most common eye color.",
EYE_COLOR//HAZEL,"Hazel eyes. These are uncommon",
HR,"Heart Rate",LOINC/8867-4
TEMP,"Body Temperature",LOINC/8310-5
"""

STAGE_CONFIG_YAML = """
filter_patients:
  min_events_per_patient: 5
add_time_derived_measurements:
  age:
    DOB_code: "DOB"
    age_code: "AGE"
    age_unit: "years"
fit_outlier_detection:
  aggregations:
    - "values/n_occurrences"
    - "values/sum"
    - "values/sum_sqd"
occlude_outliers:
  stddev_cutoff: 1
fit_normalization:
  aggregations:
    - "code/n_occurrences"
    - "code/n_patients"
    - "values/n_occurrences"
    - "values/sum"
    - "values/sum_sqd"
"""

# After filtering out patients with fewer than 5 events:
POST_FILTER_YAML = parse_shards_yaml("""
  "filter_patients/train/0": |-2
    patient_id,time,code,numeric_value
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
  "filter_patients/train/1": |-2
    patient_id,time,code,numeric_value
  "filter_patients/tuning/0": |-2
    patient_id,time,code,numeric_value
  "filter_patients/held_out/0": |-2
    patient_id,time,code,numeric_value
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
""")

WANT_NRTs = {
    "train/1.nrt": JointNestedRaggedTensorDict({}),  # this shard was fully filtered out.
    "tuning/0.nrt": JointNestedRaggedTensorDict({}),  # this shard was fully filtered out.
}


def test_pipeline():
    multi_stage_transform_tester(
        transform_scripts=[
            FILTER_PATIENTS_SCRIPT,
            ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT,
            AGGREGATE_CODE_METADATA_SCRIPT,
            OCCLUDE_OUTLIERS_SCRIPT,
            AGGREGATE_CODE_METADATA_SCRIPT,
            FIT_VOCABULARY_INDICES_SCRIPT,
            NORMALIZATION_SCRIPT,
            TOKENIZATION_SCRIPT,
            TENSORIZATION_SCRIPT,
        ],
        stage_names=[
            "filter_patients",
            "add_time_derived_measurements",
            "fit_outlier_detection",
            "occlude_outliers",
            "fit_normalization",
            "fit_vocabulary_indices",
            "normalization",
            "tokenization",
            "tensorization",
        ],
        stage_configs=STAGE_CONFIG_YAML,
        want_data=WANT_NRTs,
        input_code_metadata=MEDS_CODE_METADATA_FILE,
    )
