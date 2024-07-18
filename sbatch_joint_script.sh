#!/usr/bin/env bash
#SBATCH -c 5                           # Request one core
#SBATCH -t 0-03:00                      # Runtime in D-HH:MM format
#SBATCH -p gpu                        # Partition to run in
#SBATCH --mem=40GB                     # Memory total in MiB (for all cores)

cd /mnt/nfs/home/alinoorim/MEDS_polars_functions || exit

GEMINI_MEDS_DIR="$2"

LOG_DIR="$GEMINI_MEDS_DIR/logs"

echo "Running with saving to $LOG_DIR"

mkdir -p "$LOG_DIR"

source $HOME/myenv/bin/activate 

# PATH="/home/mbm47/.conda/envs/MEDS_pipelines/bin:$PATH" \
time mprof run --include-children --exit-code --output "$LOG_DIR/mprofile.dat" \
      ./GEMINI_Example/joint_script_slurm.sh "$@" 2> "log.txt"
