#!/usr/bin/env bash
#SBATCH -c 10                           # Request one core
#SBATCH -t 0-03:00                      # Runtime in D-HH:MM format
#SBATCH -p short                        # Partition to run in
#SBATCH --mem=300GB                     # Memory total in MiB (for all cores)
#SBATCH -o MIMIC_IV_MEDS_%j_sbatch.out  # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e MIMIC_IV_MEDS_%j_sbatch.err  # File to which STDERR will be written, including job ID (%j)

cd /n/data1/hms/dbmi/zaklab/mmd/MEDS_polars_functions

MIMICIV_RAW_DIR="$1"
MIMICIV_PREMEDS_DIR="$2"
MIMICIV_MEDS_DIR="$3"
N_PARALLEL_WORKERS="$4"

LOG_DIR="$MIMICIV_MEDS_DIR/.logs"

echo "Running with saving to $LOG_DIR"

mkdir -p $LOG_DIR

PATH="/home/mbm47/.conda/envs/MEDS_pipelines/bin:$PATH" \
  time mprof run --include-children --exit-code --output "$LOG_DIR/mprofile.dat" \
      ./MIMIC-IV_Example/joint_script.sh "$@" 2> $LOG_DIR/timings.txt
