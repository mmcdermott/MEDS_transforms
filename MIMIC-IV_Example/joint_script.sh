#!/usr/bin/env bash

MIMICIV_RAW_DIR="$1"
MIMICIV_PREMEDS_DIR="$2"
MIMICIV_MEDS_DIR="$3"
N_PARALLEL_WORKERS="$4"

shift 4

echo "Running pre-MEDS conversion."
./MIMIC-IV_Example/pre_MEDS.py raw_cohort_dir=$MIMICIV_RAW_DIR output_dir=$MIMICIV_PREMEDS_DIR

echo "Running shard_events.py with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/shard_events.py \
    --multirun \
    worker="range(1,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir=$MIMICIV_PREMEDS_DIR \
    cohort_dir=$MIMICIV_MEDS_DIR \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Splitting patients in serial"
./scripts/extraction/split_and_shard_patients.py \
    input_dir=$MIMICIV_PREMEDS_DIR \
    cohort_dir=$MIMICIV_MEDS_DIR \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/convert_to_sharded_events.py \
    --multirun \
    worker="range(1,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir=$MIMICIV_PREMEDS_DIR \
    cohort_dir=$MIMICIV_MEDS_DIR \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"

echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/merge_to_MEDS_cohort.py \
    --multirun \
    worker="range(1,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    input_dir=$MIMICIV_PREMEDS_DIR \
    cohort_dir=$MIMICIV_MEDS_DIR \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"
