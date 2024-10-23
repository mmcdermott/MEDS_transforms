#!/usr/bin/env bash

# This makes the script fail if any internal script fails
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <EICU_RAW_DIR> <EICU_PREMEDS_DIR> <EICU_MEDS_DIR>"
    echo
    echo "This script processes eICU data through several steps, handling raw data conversion,"
    echo "sharding events, splitting subjects, converting to sharded events, and merging into a MEDS cohort."
    echo
    echo "Arguments:"
    echo "  EICU_RAW_DIR        Directory containing raw eICU data files."
    echo "  EICU_PREMEDS_DIR    Output directory for pre-MEDS data."
    echo "  EICU_MEDS_DIR       Output directory for processed MEDS data."
    echo "  (OPTIONAL) do_unzip=true OR do_unzip=false     Optional flag to unzip files before processing."
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

EICU_RAW_DIR="$1"
EICU_PRE_MEDS_DIR="$2"
EICU_MEDS_COHORT_DIR="$3"

export EICU_PRE_MEDS_DIR="$2"
export EICU_MEDS_COHORT_DIR="$3"

shift 4

echo "Note that eICU has a lot more observations per subject than does MIMIC-IV, so to keep to a reasonable "
echo "memory burden (e.g., < 150GB per worker), you will want a smaller shard size, as well as to turn off "
echo "the final unique check (which should not be necessary given the structure of eICU and is expensive) "
echo "in the merge stage. You can do this by setting the following parameters at the end of the mandatory "
echo "args when running this script:"
echo "  * stage_configs.split_and_shard_subjects.n_subjects_per_shard=10000"
echo "  * stage_configs.merge_to_MEDS_cohort.unique_by=null"
echo "Additionally, consider reducing N_PARALLEL_WORKERS if > 1"

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

if [ -n "$_DO_UNZIP_ARG_STR" ]; then
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
EVENT_CONVERSION_CONFIG_FP="$(pwd)/configs/event_configs.yaml"
PIPELINE_CONFIG_FP="$(pwd)/configs/extract_eICU.yaml"
PRE_MEDS_PY_FP="$(pwd)/pre_MEDS.py"

# We export these variables separately from their assignment so that any errors during assignment are caught.
export EVENT_CONVERSION_CONFIG_FP
export PIPELINE_CONFIG_FP
export PRE_MEDS_PY_FP


if [ "$DO_UNZIP" == "true" ]; then
  GZ_FILES="${EICU_RAW_DIR}/*.csv.gz"
  if compgen -G "$GZ_FILES" > /dev/null; then
    echo "Unzipping csv.gz files matching $GZ_FILES."
    for file in $GZ_FILES; do gzip -d --force "$file"; done
  else
    echo "No csz.gz files to unzip at $GZ_FILES."
  fi
else
  echo "Skipping unzipping."
fi

echo "Running pre-MEDS conversion."
./eICU_Example/pre_MEDS.py raw_cohort_dir="$EICU_RAW_DIR" output_dir="$EICU_PREMEDS_DIR"

if [ -z "$N_WORKERS" ]; then
  echo "Setting N_WORKERS to 1 to avoid issues with the runners."
  export N_WORKERS="1"
fi

echo "Running extraction pipeline."
MEDS_transform-runner "pipeline_config_fp=$PIPELINE_CONFIG_FP" "$@"
