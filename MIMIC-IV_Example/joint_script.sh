#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <MIMICIV_RAW_DIR> <MIMICIV_PREMEDS_DIR> <MIMICIV_MEDS_DIR> <N_PARALLEL_WORKERS>"
    echo
    echo "This script processes MIMIC-IV data through several steps, handling raw data conversion,"
    echo "sharding events, splitting patients, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  MIMICIV_RAW_DIR        Directory containing raw MIMIC-IV data files."
    echo "  MIMICIV_PREMEDS_DIR    Output directory for pre-MEDS data."
    echo "  MIMICIV_MEDS_DIR       Output directory for processed MEDS data."
    echo "  N_PARALLEL_WORKERS     Number of parallel workers for processing."
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
if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

MIMICIV_RAW_DIR="$1"
MIMICIV_PREMEDS_DIR="$2"
MIMICIV_MEDS_DIR="$3"
N_PARALLEL_WORKERS="$4"

shift 4

echo "Running pre-MEDS conversion."
./MIMIC-IV_Example/pre_MEDS.py raw_cohort_dir="$MIMICIV_RAW_DIR" output_dir="$MIMICIV_PREMEDS_DIR"

echo "Running shard_events.py with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/shard_events.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Splitting patients in serial"
./scripts/extraction/split_and_shard_patients.py \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/convert_to_sharded_events.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/merge_to_MEDS_cohort.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"
