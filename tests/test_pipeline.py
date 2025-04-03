"""Tests a multi-stage pre-processing pipeline via the Runner utility. Only checks final outputs.

In this test, the following stages are run:
  - filter_subjects
  - add_time_derived_measurements
  - fit_outlier_detection
  - occlude_outliers
  - fit_normalization
  - fit_vocabulary_indices
  - normalization

The stage configuration arguments will be as given in the yaml block below:
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from MEDS_transforms.pytest_plugin import test_pipeline

RUNNER_SCRIPT = "MEDS_transform-pipeline"

PIPELINE_YAML = """
defaults:
  - _preprocess
  - _self_

input_dir: {input_dir}
cohort_dir: {cohort_dir}

description: "A test pipeline for the MEDS-transforms pipeline runner."

stages:
  - filter_subjects
  - add_time_derived_measurements
  - fit_outlier_detection
  - occlude_outliers
  - fit_normalization
  - fit_vocabulary_indices
  - normalization

stage_configs:
  filter_subjects:
    min_events_per_subject: 5
  add_time_derived_measurements:
    age:
      DOB_code: "DOB"
      age_code: "AGE"
      age_unit: "years"
    time_of_day:
      time_of_day_code: "TIME_OF_DAY"
      endpoints: [6, 12, 18, 24]
  fit_outlier_detection:
    _base_stage: "aggregate_code_metadata"
    aggregations:
      - "values/n_occurrences"
      - "values/sum"
      - "values/sum_sqd"
  occlude_outliers:
    stddev_cutoff: 1
  fit_normalization:
    _base_stage: "aggregate_code_metadata"
    aggregations:
      - "code/n_occurrences"
      - "code/n_subjects"
      - "values/n_occurrences"
      - "values/sum"
      - "values/sum_sqd"
"""

STAGE_SCENARIO_SEQUENCE = [
    "filter_subjects",
    "add_time_derived_measurements/in_example_pipeline",
    "aggregate_code_metadata/in_example_pipeline/fit_outlier_detection",
    "occlude_outliers/in_example_pipeline",
    "aggregate_code_metadata/in_example_pipeline/fit_normalization",
    "fit_vocabulary_indices/in_example_pipeline",
    "normalization/in_example_pipeline",
]


# Normally, you wouldn't need to specify all of these scripts, but in testing with local scripts we need to
# specify them all as they need to point to their python paths.
STAGE_RUNNER_YAML = """
fit_normalization:
  _base_stage: "aggregate_code_metadata"
"""

PARALLEL_STAGE_RUNNER_YAML = f"""
parallelize:
  n_workers: 2
  launcher: "joblib"

{STAGE_RUNNER_YAML}
"""


def test_example_pipeline():
    test_pipeline(PIPELINE_YAML, STAGE_RUNNER_YAML, STAGE_SCENARIO_SEQUENCE)


@pytest.mark.parallelized
def test_example_pipeline_parallel():
    test_pipeline(PIPELINE_YAML, PARALLEL_STAGE_RUNNER_YAML, STAGE_SCENARIO_SEQUENCE)


NO_ARGS_HELP_STR = """
== MEDS-Transforms Pipeline Runner ==
MEDS-Transforms Pipeline Runner is a command line tool for running entire MEDS-transform pipelines in a single
command.

**MEDS-transforms Pipeline description:**

No description provided.
"""

WITH_CONFIG_HELP_STR = """
== MEDS-Transforms Pipeline Runner ==
MEDS-Transforms Pipeline Runner is a command line tool for running entire MEDS-transform pipelines in a single
command.

**MEDS-transforms Pipeline description:**

A test pipeline for the MEDS-transforms pipeline runner.
"""


def test_pipeline_help():
    out = subprocess.run(f"{RUNNER_SCRIPT} -h", shell=True, check=True, capture_output=True)
    assert NO_ARGS_HELP_STR.strip() == out.stdout.decode("utf-8").strip()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        cohort_dir = Path(tmpdir) / "cohort"

        pipeline_str = PIPELINE_YAML.format(input_dir=input_dir, cohort_dir=cohort_dir)

        pipeline_fp = Path(tmpdir) / "pipeline.yaml"
        pipeline_fp.write_text(pipeline_str)

        out = subprocess.run(
            f"{RUNNER_SCRIPT} -h pipeline_config_fp={pipeline_fp}",
            shell=True,
            check=True,
            capture_output=True,
        )
        assert WITH_CONFIG_HELP_STR.strip() == out.stdout.decode("utf-8").strip()
