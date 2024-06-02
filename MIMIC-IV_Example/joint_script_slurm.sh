#!/usr/bin/env bash

MIMICIV_RAW_DIR="$1"
MIMICIV_PREMEDS_DIR="$2"
MIMICIV_MEDS_DIR="$3"
N_PARALLEL_WORKERS="$4"

shift 4

# Note we use `--multirun` throughout here due to ensure the submitit launcher is used throughout, so that
# this doesn't fall back on running anything locally in a setting where only slurm worker nodes have
# sufficient computational resources to run the actual jobs.

echo "Running pre-MEDS conversion on one worker."
./MIMIC-IV_Example/pre_MEDS.py \
  --multirun \
  worker="range(0,1)" \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=60 \
  hydra.launcher.cpus_per_task=10 \
  hydra.launcher.mem_gb=50 \
  hydra.launcher.partition="short" \
  raw_cohort_dir="$MIMICIV_RAW_DIR" \
  output_dir="$MIMICIV_PREMEDS_DIR" || exit 1

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
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml || exit 1

echo "Splitting patients on one worker"
./scripts/extraction/split_and_shard_patients.py \
    --multirun \
    worker="range(0,1)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.mem_gb=50 \
    hydra.launcher.partition="short" \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@" || exit 1

echo "Converting to sharded events with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/convert_to_sharded_events.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.mem_gb=50 \
    hydra.launcher.partition="short" \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@" || exit 1

echo "Merging to a MEDS cohort with $N_PARALLEL_WORKERS workers in parallel"
./scripts/extraction/merge_to_MEDS_cohort.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.mem_gb=50 \
    hydra.launcher.partition="short" \
    input_dir="$MIMICIV_PREMEDS_DIR" \
    cohort_dir="$MIMICIV_MEDS_DIR" \
    event_conversion_config_fp=./MIMIC-IV_Example/configs/event_configs.yaml "$@"
