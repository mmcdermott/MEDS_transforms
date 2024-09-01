"""Tests a multi-stage pre-processing pipeline via the Runner utility. Only checks final outputs.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.

In this test, the following stages are run:
  - filter_subjects
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


from functools import partial

from meds import code_metadata_filepath, subject_splits_filepath

from tests import RUNNER_SCRIPT, USE_LOCAL_SCRIPTS
from tests.MEDS_Transforms import (
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT,
    AGGREGATE_CODE_METADATA_SCRIPT,
    FILTER_SUBJECTS_SCRIPT,
    FIT_VOCABULARY_INDICES_SCRIPT,
    NORMALIZATION_SCRIPT,
    OCCLUDE_OUTLIERS_SCRIPT,
    TOKENIZATION_SCRIPT,
)
from tests.MEDS_Transforms.test_multi_stage_preprocess_pipeline import (
    MEDS_CODE_METADATA,
    WANT_FILTER,
    WANT_FIT_NORMALIZATION,
    WANT_FIT_OUTLIERS,
    WANT_FIT_VOCABULARY_INDICES,
    WANT_NORMALIZATION,
    WANT_OCCLUDE_OUTLIERS,
    WANT_TIME_DERIVED,
    WANT_TOKENIZATION_EVENT_SEQS,
    WANT_TOKENIZATION_SCHEMAS,
    WANT_NRTs,
)
from tests.MEDS_Transforms.transform_tester_base import MEDS_SHARDS, SPLITS_DF
from tests.utils import add_params, exact_str_regex, single_stage_tester

# Normally, you wouldn't need to specify all of these scripts, but in testing with local scripts we need to
# specify them all as they need to point to their python paths.
if USE_LOCAL_SCRIPTS:
    STAGE_RUNNER_YAML = f"""
filter_subjects:
  script: "python {FILTER_SUBJECTS_SCRIPT}"

add_time_derived_measurements:
  script: "python {ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT}"

occlude_outliers:
  script: "python {OCCLUDE_OUTLIERS_SCRIPT}"

fit_normalization:
  script: "python {AGGREGATE_CODE_METADATA_SCRIPT}"

fit_vocabulary_indices:
  script: "python {FIT_VOCABULARY_INDICES_SCRIPT}"

normalization:
  script: "python {NORMALIZATION_SCRIPT}"

tokenization:
  script: "python {TOKENIZATION_SCRIPT}"
    """
else:
    STAGE_RUNNER_YAML = f"""
fit_normalization:
  script: {AGGREGATE_CODE_METADATA_SCRIPT}
    """

PARALLEL_STAGE_RUNNER_YAML = f"""
parallelize:
  n_workers: 2
  launcher: "joblib"

{STAGE_RUNNER_YAML}
"""


PIPELINE_YAML = f"""
defaults:
  - _preprocess
  - _self_

input_dir: {{input_dir}}
cohort_dir: {{cohort_dir}}

description: "A test pipeline for the MEDS-transforms pipeline runner."

stages:
  - filter_subjects
  - add_time_derived_measurements
  - fit_outlier_detection
  - occlude_outliers
  - fit_normalization
  - fit_vocabulary_indices
  - normalization
  - tokenization
  - tensorization

stage_configs:
  filter_subjects:
    min_events_per_subject: 5
  add_time_derived_measurements:
    age:
      DOB_code: "DOB" # This is the MEDS official code for BIRTH
      age_code: "AGE"
      age_unit: "years"
    time_of_day:
      time_of_day_code: "TIME_OF_DAY"
      endpoints: [6, 12, 18, 24]
  fit_outlier_detection:
    _script: {("python " if USE_LOCAL_SCRIPTS else "") + str(AGGREGATE_CODE_METADATA_SCRIPT)}
    aggregations:
      - "values/n_occurrences"
      - "values/sum"
      - "values/sum_sqd"
  occlude_outliers:
    stddev_cutoff: 1
  fit_normalization:
    aggregations:
      - "code/n_occurrences"
      - "code/n_subjects"
      - "values/n_occurrences"
      - "values/sum"
      - "values/sum_sqd"
"""

NO_ARGS_HELP_STR = """
== MEDS-Transforms Pipeline Runner ==
MEDS-Transforms Pipeline Runner is a command line tool for running entire MEDS-transform pipelines in a single
command.

Runs the entire pipeline, end-to-end, based on the configuration provided.

This script will launch many subsidiary commands via `subprocess`, one for each stage of the specified
pipeline.

**MEDS-transforms Pipeline description:**

No description provided.
"""

WITH_CONFIG_HELP_STR = """
== MEDS-Transforms Pipeline Runner ==
MEDS-Transforms Pipeline Runner is a command line tool for running entire MEDS-transform pipelines in a single
command.

Runs the entire pipeline, end-to-end, based on the configuration provided.

This script will launch many subsidiary commands via `subprocess`, one for each stage of the specified
pipeline.

**MEDS-transforms Pipeline description:**

A test pipeline for the MEDS-transforms pipeline runner.
"""


def test_pipeline():
    single_stage_tester(
        script=str(RUNNER_SCRIPT) + " -h",
        config_name="runner",
        stage_name=None,
        stage_kwargs=None,
        do_pass_stage_name=False,
        do_use_config_yaml=False,
        input_files={},
        want_outputs={},
        assert_no_other_outputs=True,
        should_error=False,
        test_name="Runner Help Test",
        do_include_dirs=False,
        hydra_verbose=False,
        stdout_regex=exact_str_regex(NO_ARGS_HELP_STR.strip()),
    )

    single_stage_tester(
        script=str(RUNNER_SCRIPT) + " -h",
        config_name="runner",
        stage_name=None,
        stage_kwargs=None,
        do_pass_stage_name=False,
        do_use_config_yaml=False,
        input_files={"pipeline.yaml": partial(add_params, PIPELINE_YAML)},
        want_outputs={},
        assert_no_other_outputs=True,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        test_name="Runner Help Test",
        do_include_dirs=False,
        hydra_verbose=False,
        stdout_regex=exact_str_regex(WITH_CONFIG_HELP_STR.strip()),
    )

    single_stage_tester(
        script=RUNNER_SCRIPT,
        config_name="runner",
        stage_name=None,
        stage_kwargs=None,
        do_pass_stage_name=False,
        do_use_config_yaml=False,
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "pipeline.yaml": partial(add_params, PIPELINE_YAML),
            "stage_runner.yaml": STAGE_RUNNER_YAML,
        },
        want_outputs={
            **WANT_FIT_NORMALIZATION,
            **WANT_FIT_OUTLIERS,
            **WANT_FIT_VOCABULARY_INDICES,
            **WANT_FILTER,
            **WANT_TIME_DERIVED,
            **WANT_OCCLUDE_OUTLIERS,
            **WANT_NORMALIZATION,
            **WANT_TOKENIZATION_SCHEMAS,
            **WANT_TOKENIZATION_EVENT_SEQS,
            **WANT_NRTs,
        },
        assert_no_other_outputs=False,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        stage_runner_fp="{input_dir}/stage_runner.yaml",
        test_name="Runner Test",
        do_include_dirs=False,
        df_check_kwargs={"check_column_order": False},
    )

    single_stage_tester(
        script=RUNNER_SCRIPT,
        config_name="runner",
        stage_name=None,
        stage_kwargs=None,
        do_pass_stage_name=False,
        do_use_config_yaml=False,
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "pipeline.yaml": partial(add_params, PIPELINE_YAML),
            "stage_runner.yaml": PARALLEL_STAGE_RUNNER_YAML,
        },
        want_outputs={
            **WANT_FIT_NORMALIZATION,
            **WANT_FIT_OUTLIERS,
            **WANT_FIT_VOCABULARY_INDICES,
            **WANT_FILTER,
            **WANT_TIME_DERIVED,
            **WANT_OCCLUDE_OUTLIERS,
            **WANT_NORMALIZATION,
            **WANT_TOKENIZATION_SCHEMAS,
            **WANT_TOKENIZATION_EVENT_SEQS,
            **WANT_NRTs,
        },
        assert_no_other_outputs=False,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        stage_runner_fp="{input_dir}/stage_runner.yaml",
        test_name="Runner Test with parallelism",
        do_include_dirs=False,
        df_check_kwargs={"check_column_order": False},
    )
