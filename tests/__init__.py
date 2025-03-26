import os

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

code_root = root / "src" / "MEDS_transforms"
transforms_root = code_root / "transforms"
filters_root = code_root / "filters"

USE_LOCAL_SCRIPTS = os.environ.get("DO_USE_LOCAL_SCRIPTS", "0") == "1"

if USE_LOCAL_SCRIPTS:
    # Runner
    RUNNER_SCRIPT = code_root / "runner.py"

    # Root Source
    AGGREGATE_CODE_METADATA_SCRIPT = code_root / "aggregate_code_metadata.py"
    FIT_VOCABULARY_INDICES_SCRIPT = code_root / "fit_vocabulary_indices.py"
    RESHARD_TO_SPLIT_SCRIPT = code_root / "reshard_to_split.py"

    # Filters
    FILTER_MEASUREMENTS_SCRIPT = filters_root / "filter_measurements.py"
    FILTER_SUBJECTS_SCRIPT = filters_root / "filter_subjects.py"

    # Transforms
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT = transforms_root / "add_time_derived_measurements.py"
    REORDER_MEASUREMENTS_SCRIPT = transforms_root / "reorder_measurements.py"
    EXTRACT_VALUES_SCRIPT = transforms_root / "extract_values.py"
    NORMALIZATION_SCRIPT = transforms_root / "normalization.py"
    OCCLUDE_OUTLIERS_SCRIPT = transforms_root / "occlude_outliers.py"
else:
    # Runner
    RUNNER_SCRIPT = "MEDS_transform-runner"

    # Root Source
    AGGREGATE_CODE_METADATA_SCRIPT = "MEDS_transform-aggregate_code_metadata"
    FIT_VOCABULARY_INDICES_SCRIPT = "MEDS_transform-fit_vocabulary_indices"
    RESHARD_TO_SPLIT_SCRIPT = "MEDS_transform-reshard_to_split"

    # Filters
    FILTER_MEASUREMENTS_SCRIPT = "MEDS_transform-filter_measurements"
    FILTER_SUBJECTS_SCRIPT = "MEDS_transform-filter_subjects"

    # Transforms
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT = "MEDS_transform-add_time_derived_measurements"
    REORDER_MEASUREMENTS_SCRIPT = "MEDS_transform-reorder_measurements"
    EXTRACT_VALUES_SCRIPT = "MEDS_transform-extract_values"
    NORMALIZATION_SCRIPT = "MEDS_transform-normalization"
    OCCLUDE_OUTLIERS_SCRIPT = "MEDS_transform-occlude_outliers"
