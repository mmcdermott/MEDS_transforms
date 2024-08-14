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
)

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
    )
