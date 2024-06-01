#!/usr/bin/env bash

EICU_RAW_DIR="$1"
EICU_PREMEDS_DIR="$2"
EICU_MEDS_DIR="$3"
N_PARALLEL_WORKERS="$4"

shift 4

echo "Running pre-MEDS conversion."
./eICU_Example/pre_MEDS.py raw_cohort_dir="$EICU_RAW_DIR" output_dir="$EICU_PREMEDS_DIR"

echo "Running shard_events.py with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/shard_events.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$EICU_PREMEDS_DIR" \
    cohort_dir="$EICU_MEDS_DIR" \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml "$@"

echo "Splitting patients in serial"
./scripts/extraction/split_and_shard_patients.py \
    input_dir="$EICU_PREMEDS_DIR" \
    cohort_dir="$EICU_MEDS_DIR" \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml "$@"

echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/convert_to_sharded_events.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$EICU_PREMEDS_DIR" \
    cohort_dir="$EICU_MEDS_DIR" \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml "$@"

echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/merge_to_MEDS_cohort.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir="$EICU_PREMEDS_DIR" \
    cohort_dir="$EICU_MEDS_DIR" \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml "$@"
