#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <EICU_RAW_DIR> <EICU_PREMEDS_DIR> <EICU_MEDS_DIR> <N_PARALLEL_WORKERS>"
    echo
    echo "This script processes eICU data through several steps, handling raw data conversion,"
    echo "sharding events, splitting subjects, converting to sharded events, and merging into a MEDS cohort."
    echo "This script uses slurm to process the data in parallel via the 'submitit' Hydra launcher."
    echo
    echo "Arguments:"
    echo "  EICU_RAW_DIR        Directory containing raw eICU data files."
    echo "  EICU_PREMEDS_DIR    Output directory for pre-MEDS data."
    echo "  EICU_MEDS_DIR       Output directory for processed MEDS data."
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
if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

EICU_RAW_DIR="$1"
EICU_PREMEDS_DIR="$2"
EICU_MEDS_DIR="$3"
N_PARALLEL_WORKERS="$4"

shift 4

# Note we use `--multirun` throughout here due to ensure the submitit launcher is used throughout, so that
# this doesn't fall back on running anything locally in a setting where only slurm worker nodes have
# sufficient computational resources to run the actual jobs.

echo "Running pre-MEDS conversion on one worker."
./eICU_Example/pre_MEDS.py \
  --multirun \
  worker="range(0,1)" \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=60 \
  hydra.launcher.cpus_per_task=10 \
  hydra.launcher.mem_gb=50 \
  hydra.launcher.partition="short" \
  raw_cohort_dir="$EICU_RAW_DIR" \
  output_dir="$EICU_PREMEDS_DIR"

echo "Trying submitit launching with $N_PARALLEL_WORKERS jobs."

./scripts/extraction/shard_events.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.mem_gb=50 \
    hydra.launcher.partition="short" \
    "hydra.job.env_copy=[PATH]" \
    input_dir="$EICU_PREMEDS_DIR" \
    cohort_dir="$EICU_MEDS_DIR" \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml

echo "Splitting subjects on one worker"
./scripts/extraction/split_and_shard_subjects.py \
    --multirun \
    worker="range(0,1)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.mem_gb=50 \
    hydra.launcher.partition="short" \
    input_dir="$EICU_PREMEDS_DIR" \
    cohort_dir="$EICU_MEDS_DIR" \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml "$@"

echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/convert_to_sharded_events.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.mem_gb=50 \
    hydra.launcher.partition="short" \
    input_dir="$EICU_PREMEDS_DIR" \
    cohort_dir="$EICU_MEDS_DIR" \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml "$@"

echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/merge_to_MEDS_cohort.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.mem_gb=50 \
    hydra.launcher.partition="short" \
    input_dir="$EICU_PREMEDS_DIR" \
    cohort_dir="$EICU_MEDS_DIR" \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml "$@"
