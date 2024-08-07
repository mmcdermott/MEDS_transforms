#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <AUMC_RAW_DIR> <AUMC_PREMEDS_DIR> <AUMC_MEDS_DIR> <N_PARALLEL_WORKERS>"
    echo
    echo "This script processes AUMCdb data through several steps, handling raw data conversion,"
    echo "sharding events, splitting patients, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  AUMC_RAW_DIR        Directory containing raw AUMCdb data files."
    echo "  AUMC_PREMEDS_DIR    Output directory for pre-MEDS data."
    echo "  AUMC_MEDS_DIR       Output directory for processed MEDS data."
    echo "  N_PARALLEL_WORKERS  Number of parallel workers for processing."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

# Check if the first parameter is '-h' or '--help'
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters
if [ "$#" -lt 4 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

AUMC_RAW_DIR="$1"
AUMC_PREMEDS_DIR="$2"
AUMC_MEDS_DIR="$3"
N_PARALLEL_WORKERS="$4"

shift 4

echo "Note that AUMCdb has a lot of observations in the numericitems, so to keep to a reasonable "
echo "memory burden (e.g., < 150GB per worker), you will want a smaller shard size, as well as to turn off "
echo "the final unique check (which should not be necessary given the structure of AUMCdb and is expensive) "
echo "in the merge stage. You can do this by setting the following parameters at the end of the mandatory "
echo "args when running this script:"
echo "  * stage_configs.split_and_shard_patients.n_patients_per_shard=10000"
echo "  * stage_configs.merge_to_MEDS_cohort.unique_by=null"


echo "Running pre-MEDS conversion."
./AUMC_Example/pre_MEDS.py raw_cohort_dir="$AUMC_RAW_DIR" output_dir="$AUMC_PREMEDS_DIR"

echo "Running shard_events.py with $N_PARALLEL_WORKERS workers in parallel"
MEDS_extract-shard_events \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$AUMC_PREMEDS_DIR" \
    cohort_dir="$AUMC_MEDS_DIR" \
    stage="shard_events" \
    event_conversion_config_fp=./AUMC_Example/configs/event_configs.yaml "$@"

echo "Splitting patients in serial"
MEDS_extract-split_and_shard_patients \
    input_dir="$AUMC_PREMEDS_DIR" \
    cohort_dir="$AUMC_MEDS_DIR" \
    stage="split_and_shard_patients" \
    stage_configs.split_and_shard_patients.n_patients_per_shard=10000 \
    event_conversion_config_fp=./AUMC_Example/configs/event_configs.yaml "$@"

echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
MEDS_extract-convert_to_sharded_events \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$AUMC_PREMEDS_DIR" \
    cohort_dir="$AUMC_MEDS_DIR" \
    stage="convert_to_sharded_events" \
    event_conversion_config_fp=./AUMC_Example/configs/event_configs.yaml "$@"

echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
MEDS_extract-merge_to_MEDS_cohort \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$AUMC_PREMEDS_DIR" \
    cohort_dir="$AUMC_MEDS_DIR" \
    stage="merge_to_MEDS_cohort" \
    event_conversion_config_fp=./AUMC_Example/configs/event_configs.yaml "$@"

echo "Finalizing MEDS data with $N_PARALLEL_WORKERS workers in parallel"
MEDS_extract-finalize_MEDS_data \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$AUMC_RAW_DIR" \
    cohort_dir="$AUMC_MEDS_DIR" \
    stage="finalize_MEDS_data" \
    etl_metadata.dataset_name="AUMCdb" \
    etl_metadata.dataset_version="1.0.2" \
    event_conversion_config_fp=./AUMC_Example/configs/event_configs.yaml "$@"