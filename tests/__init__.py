# Runner
RUNNER_SCRIPT = "MEDS_transform-pipeline"

# Stages
__stage_pattern = "MEDS_transform-stage __null__ {stage_name}"

AGGREGATE_CODE_METADATA_SCRIPT = __stage_pattern.format(stage_name="aggregate_code_metadata")
FIT_VOCABULARY_INDICES_SCRIPT = __stage_pattern.format(stage_name="fit_vocabulary_indices")
RESHARD_TO_SPLIT_SCRIPT = __stage_pattern.format(stage_name="reshard_to_split")
FILTER_MEASUREMENTS_SCRIPT = __stage_pattern.format(stage_name="filter_measurements")
FILTER_SUBJECTS_SCRIPT = __stage_pattern.format(stage_name="filter_subjects")
ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT = __stage_pattern.format(stage_name="add_time_derived_measurements")
REORDER_MEASUREMENTS_SCRIPT = __stage_pattern.format(stage_name="reorder_measurements")
EXTRACT_VALUES_SCRIPT = __stage_pattern.format(stage_name="extract_values")
NORMALIZATION_SCRIPT = __stage_pattern.format(stage_name="normalization")
OCCLUDE_OUTLIERS_SCRIPT = __stage_pattern.format(stage_name="occlude_outliers")
