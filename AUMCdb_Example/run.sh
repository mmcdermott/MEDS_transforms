#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <AUMC_RAW_DIR> <AUMC_PRE_MEDS_DIR> <AUMC_MEDS_COHORT_DIR>"
    echo
    echo "This script processes the AUMCdb (AmsterdamUMCdb, Amsterdam University Medical Center database, short version: AUMC) data through several steps,"
    echo "handling raw data conversion, sharding events, splitting subjects, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  AUMC_RAW_DIR                                Directory containing raw AUMCdb data files."
    echo "  AUMC_PREMEDS_DIR                            Output directory for pre-MEDS data."
    echo "  AUMC_MEDS_DIR                               Output directory for processed MEDS data."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

echo "Unsetting SLURM_CPU_BIND in case you're running this on a slurm interactive node with slurm parallelism"
unset SLURM_CPU_BIND

# Check if the first parameter is '-h' or '--help'
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters
if [ "$#" -lt 3 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

export AUMC_RAW_DIR=$1
export AUMC_PRE_MEDS_DIR=$2
export AUMC_MEDS_COHORT_DIR=$3
shift 3

# TODO: Add wget blocks once testing is validated.

EVENT_CONVERSION_CONFIG_FP="$(pwd)/configs/event_configs.yaml"
PIPELINE_CONFIG_FP="$(pwd)/configs/extract_AUMC.yaml"
PRE_MEDS_PY_FP="$(pwd)/pre_MEDS.py"

# We export these variables separately from their assignment so that any errors during assignment are caught.
export EVENT_CONVERSION_CONFIG_FP
export PIPELINE_CONFIG_FP
export PRE_MEDS_PY_FP


echo "Running pre-MEDS conversion."
python "$PRE_MEDS_PY_FP" input_dir="$AUMC_RAW_DIR" cohort_dir="$AUMC_PRE_MEDS_DIR"

if [ -z "$N_WORKERS" ]; then
  echo "Setting N_WORKERS to 1 to avoid issues with the runners."
  export N_WORKERS="1"
fi

echo "Running extraction pipeline."
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP" "$@"
