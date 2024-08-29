#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <MIMICIV_RAW_DIR> <MIMICIV_PRE_MEDS_DIR> <MIMICIV_MEDS_COHORT_DIR>"
    echo
    echo "This script processes MIMIC-IV data through several steps, handling raw data conversion,"
    echo "sharding events, splitting subjects, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  MIMICIV_RAW_DIR                                Directory containing raw MIMIC-IV data files."
    echo "  MIMICIV_PREMEDS_DIR                            Output directory for pre-MEDS data."
    echo "  MIMICIV_MEDS_DIR                               Output directory for processed MEDS data."
    echo "  (OPTIONAL) do_unzip=true OR do_unzip=false     Optional flag to unzip files before processing."
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

export MIMICIV_RAW_DIR=$1
export MIMICIV_PRE_MEDS_DIR=$2
export MIMICIV_MEDS_COHORT_DIR=$3
shift 3

# Defaults
_DO_UNZIP_ARG_STR=""

if [ $# -ge 1 ]; then
  case "$1" in
    do_unzip=*)
      _DO_UNZIP_ARG_STR="$1"
      shift 1
      ;;
  esac
fi

DO_UNZIP="false"

if [ ! -z "$_DO_UNZIP_ARG_STR" ]; then
  case "$_DO_UNZIP_ARG_STR" in
    do_unzip=true)
      DO_UNZIP="true"
      ;;
    do_unzip=false)
      DO_UNZIP="false"
      ;;
    *)
      echo "Error: Invalid do_unzip value. Use 'do_unzip=true' or 'do_unzip=false'."
      exit 1
      ;;
  esac
  echo "Setting DO_UNZIP=$DO_UNZIP"
fi

# TODO: Add wget blocks once testing is validated.

export EVENT_CONVERSION_CONFIG_FP="$(pwd)/configs/event_configs.yaml"
export PIPELINE_CONFIG_FP="$(pwd)/configs/extract_MIMIC.yaml"
export PRE_MEDS_PY_FP="$(pwd)/pre_MEDS.py"

if [ "$DO_UNZIP" == "true" ]; then
  GZ_FILES="${MIMICIV_RAW_DIR}/*/*.csv.gz"
  if compgen -G $GZ_FILES > /dev/null; then
    echo "Unzipping csv.gz files matching $GZ_FILES."
    for file in $GZ_FILES; do gzip -d --force "$file"; done
  else
    echo "No csz.gz files to unzip at $GZ_FILES."
  fi
else
  echo "Skipping unzipping."
fi

echo "Running pre-MEDS conversion."
python "$PRE_MEDS_PY_FP" input_dir="$MIMICIV_RAW_DIR" cohort_dir="$MIMICIV_PRE_MEDS_DIR"

if [ -z "$N_WORKERS" ]; then
  echo "Setting N_WORKERS to 1 to avoid issues with the runners."
  export N_WORKERS="1"
fi

echo "Running extraction pipeline."
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP" "$@"
