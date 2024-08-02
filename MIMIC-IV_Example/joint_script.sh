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
if [ "$#" -lt 4 ]; then
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
MEDS_extract-shard_events \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    stage="shard_events" \
    stage_configs.shard_events.infer_schema_length=999999999 \
    etl_metadata.dataset_name="MIMIC-IV" \
    etl_metadata.dataset_version="2.2" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Splitting patients in serial"
MEDS_extract-split_and_shard_patients \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    stage="split_and_shard_patients" \
    etl_metadata.dataset_name="MIMIC-IV" \
    etl_metadata.dataset_version="2.2" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
MEDS_extract-convert_to_sharded_events \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    stage="convert_to_sharded_events" \
    etl_metadata.dataset_name="MIMIC-IV" \
    etl_metadata.dataset_version="2.2" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
MEDS_extract-merge_to_MEDS_cohort \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    stage="merge_to_MEDS_cohort" \
    etl_metadata.dataset_name="MIMIC-IV" \
    etl_metadata.dataset_version="2.2" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Aggregating initial code stats with $N_PARALLEL_WORKERS workers in parallel"
MEDS_transform-aggregate_code_metadata \
    --config-name="extract" \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    stage="aggregate_code_metadata" \
    etl_metadata.dataset_name="MIMIC-IV" \
    etl_metadata.dataset_version="2.2" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

# TODO -- make this the pre-meds dir and have the pre-meds script symlink
echo "Collecting code metadata in serial."
MEDS_extract-extract_code_metadata \
    input_dir="$MIMICIV_RAW_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    stage="extract_code_metadata" \
    etl_metadata.dataset_name="MIMIC-IV" \
    etl_metadata.dataset_version="2.2" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Finalizing MEDS data with $N_PARALLEL_WORKERS workers in parallel"
MEDS_extract-finalize_MEDS_data \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$MIMICIV_RAW_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    stage="finalize_MEDS_data" \
    etl_metadata.dataset_name="MIMIC-IV" \
    etl_metadata.dataset_version="2.2" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Finalizing MEDS metadata in serial."
MEDS_extract-finalize_MEDS_metadata \
    input_dir="$MIMICIV_RAW_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    stage="finalize_MEDS_metadata" \
    etl_metadata.dataset_name="MIMIC-IV" \
    etl_metadata.dataset_version="2.2" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"
