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

from MEDS_transforms.pytest_plugin import pipeline_tester

RUNNER_SCRIPT = "MEDS_transform-pipeline"

PIPELINE_YAML = """
input_dir: {input_dir}
output_dir: {output_dir}

description: "A test pipeline for the MEDS-transforms pipeline runner."

stages:
  - filter_subjects:
      min_events_per_subject: 5
  - add_time_derived_measurements:
      age:
        DOB_code: "DOB"
        age_code: "AGE"
        age_unit: "years"
      time_of_day:
        time_of_day_code: "TIME_OF_DAY"
        endpoints: [6, 12, 18, 24]
  - fit_outlier_detection:
      _base_stage: "aggregate_code_metadata"
      aggregations:
        - "values/n_occurrences"
        - "values/sum"
        - "values/sum_sqd"
  - occlude_outliers:
      stddev_cutoff: 1
  - fit_normalization:
      _base_stage: "aggregate_code_metadata"
      aggregations:
        - "code/n_occurrences"
        - "code/n_subjects"
        - "values/n_occurrences"
        - "values/sum"
        - "values/sum_sqd"
  - fit_vocabulary_indices
  - normalization
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
PARALLEL_STAGE_RUNNER_YAML = """
parallelize:
  n_workers: 2
  launcher: "joblib"
"""


def test_example_pipeline():
    pipeline_tester(PIPELINE_YAML, None, STAGE_SCENARIO_SEQUENCE)


@pytest.mark.parallelized
def test_example_pipeline_parallel():
    pipeline_tester(PIPELINE_YAML, PARALLEL_STAGE_RUNNER_YAML, STAGE_SCENARIO_SEQUENCE)


def test_pipeline_help():
    out = subprocess.run(f"{RUNNER_SCRIPT} -h", shell=True, check=True, capture_output=True)
    help_text = out.stdout.decode("utf-8")
    assert "usage:" in help_text.lower()
    assert "pipeline_config_fp" in help_text


def test_pipeline_runner_with_done_file():
    """Test that the pipeline runner does nothing when a global done file is present at the start."""

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        log_dir = output_dir / ".logs"

        global_done_file = log_dir / "_all_stages.done"
        global_done_file.parent.mkdir(parents=True, exist_ok=True)
        global_done_file.touch()

        pipeline_config = PIPELINE_YAML.format(input_dir=input_dir, output_dir=output_dir)

        pipeline_config_path = root / "pipeline_config.yaml"
        with open(pipeline_config_path, "w") as f:
            f.write(pipeline_config)

        # Run the pipeline
        out = subprocess.run(
            f"{RUNNER_SCRIPT} {pipeline_config_path!s}", shell=True, check=False, capture_output=True
        )

        stdout = out.stdout.decode("utf-8")
        stderr = out.stderr.decode("utf-8")

        assert out.returncode == 0, f"Error running pipeline:\n{stdout}\n{stderr}"

        want_txt = "All stages are already complete. Exiting."

        pipeline_log = log_dir / "pipeline.log"
        assert pipeline_log.exists(), "Pipeline log file does not exist."
        assert pipeline_log.is_file(), "Pipeline log is not a file."
        assert want_txt in pipeline_log.read_text(), "Pipeline log does not contain expected text."


PIPELINE_YAML_NO_OUTPUT = """
input_dir: {input_dir}
output_dir: ???

description: "A test pipeline for the MEDS-transforms pipeline runner."

stages:
  - filter_subjects:
      min_events_per_subject: 5
  - add_time_derived_measurements:
      age:
        DOB_code: "DOB"
        age_code: "AGE"
        age_unit: "years"
      time_of_day:
        time_of_day_code: "TIME_OF_DAY"
        endpoints: [6, 12, 18, 24]
  - fit_outlier_detection:
      _base_stage: "aggregate_code_metadata"
      aggregations:
        - "values/n_occurrences"
        - "values/sum"
        - "values/sum_sqd"
  - occlude_outliers:
      stddev_cutoff: 1
  - fit_normalization:
      _base_stage: "aggregate_code_metadata"
      aggregations:
        - "code/n_occurrences"
        - "code/n_subjects"
        - "values/n_occurrences"
        - "values/sum"
        - "values/sum_sqd"
  - fit_vocabulary_indices
  - normalization
"""


def test_errors_without_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"

        pipeline_config = PIPELINE_YAML_NO_OUTPUT.format(input_dir=input_dir)

        pipeline_config_path = root / "pipeline_config.yaml"
        with open(pipeline_config_path, "w") as f:
            f.write(pipeline_config)

        cmd = f"{RUNNER_SCRIPT} {pipeline_config_path!s}"

        # Run the pipeline
        out = subprocess.run(cmd, shell=True, check=False, capture_output=True)

        stdout = out.stdout.decode("utf-8")
        stderr = out.stderr.decode("utf-8")

        assert out.returncode == 1, f"Pipeline should error but didn't!\n{stdout}\n{stderr}"

        want_txt = "ValueError: Pipeline configuration or override must specify an 'output_dir'"

        assert want_txt in stderr, "Pipeline did not error as expected with missing output_dir."


def test_additional_pipeline_args():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        log_dir = output_dir / ".logs"

        global_done_file = log_dir / "_all_stages.done"
        global_done_file.parent.mkdir(parents=True, exist_ok=True)
        global_done_file.touch()

        pipeline_config = PIPELINE_YAML_NO_OUTPUT.format(input_dir=input_dir)

        pipeline_config_path = root / "pipeline_config.yaml"
        with open(pipeline_config_path, "w") as f:
            f.write(pipeline_config)

        cmd = f"{RUNNER_SCRIPT} {pipeline_config_path!s} output_dir={output_dir!s}"

        # Run the pipeline
        out = subprocess.run(cmd, shell=True, check=False, capture_output=True)

        stdout = out.stdout.decode("utf-8")
        stderr = out.stderr.decode("utf-8")

        assert out.returncode == 0, f"Error running pipeline:\n{stdout}\n{stderr}"

        want_txt = "All stages are already complete. Exiting."

        pipeline_log = log_dir / "pipeline.log"
        assert pipeline_log.exists(), "Pipeline log file does not exist."
        assert pipeline_log.is_file(), "Pipeline log is not a file."
        assert want_txt in pipeline_log.read_text(), "Pipeline log does not contain expected text."
