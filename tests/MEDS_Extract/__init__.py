import os

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

extraction_root = root / "src" / "MEDS_transforms" / "extract"

if os.environ.get("DO_USE_LOCAL_SCRIPTS", "0") == "1":
    SHARD_EVENTS_SCRIPT = extraction_root / "shard_events.py"
    SPLIT_AND_SHARD_SCRIPT = extraction_root / "split_and_shard_subjects.py"
    CONVERT_TO_SHARDED_EVENTS_SCRIPT = extraction_root / "convert_to_sharded_events.py"
    MERGE_TO_MEDS_COHORT_SCRIPT = extraction_root / "merge_to_MEDS_cohort.py"
    EXTRACT_CODE_METADATA_SCRIPT = extraction_root / "extract_code_metadata.py"
    FINALIZE_DATA_SCRIPT = extraction_root / "finalize_MEDS_data.py"
    FINALIZE_METADATA_SCRIPT = extraction_root / "finalize_MEDS_metadata.py"
else:
    SHARD_EVENTS_SCRIPT = "MEDS_extract-shard_events"
    SPLIT_AND_SHARD_SCRIPT = "MEDS_extract-split_and_shard_subjects"
    CONVERT_TO_SHARDED_EVENTS_SCRIPT = "MEDS_extract-convert_to_sharded_events"
    MERGE_TO_MEDS_COHORT_SCRIPT = "MEDS_extract-merge_to_MEDS_cohort"
    EXTRACT_CODE_METADATA_SCRIPT = "MEDS_extract-extract_code_metadata"
    FINALIZE_DATA_SCRIPT = "MEDS_extract-finalize_MEDS_data"
    FINALIZE_METADATA_SCRIPT = "MEDS_extract-finalize_MEDS_metadata"
