# AUMC Example

This is an example of how to extract a MEDS dataset from AUMCdb (https://github.com/AmsterdamUMC/AmsterdamUMCdb). All scripts in this README are assumed to be run from this directory or from the directory in which the files in Step 0.5. were downloaded.

## Step 0: Installation

```bash
conda create -n MEDS python=3.12
conda activate MEDS
pip install "MEDS_transforms[local_parallelism,slurm_parallelism]"
```

If you want to profile the time and memory costs of your ETL, also install: `pip install hydra-profiler`.

## Step 0.5: Set-up

Set some environment variables and download the necessary files:

```bash
export AUMC_RAW_DIR=??? # set to the directory in which you want to store the raw data
export AUMC_PRE_MEDS_DIR=??? # set to the directory in which you want to store the intermediate MEDS data
export AUMC_MEDS_COHORT_DIR=??? # set to the directory in which you want to store the final MEDS data

export VERSION=0.0.8 # or whatever version you want
export URL="https://raw.githubusercontent.com/mmcdermott/MEDS_transforms/$VERSION/AUMC_Example"

wget $URL/run.sh
wget $URL/pre_MEDS.py
wget $URL/local_parallelism_runner.yaml
wget $URL/slurm_runner.yaml
mkdir configs
cd configs
wget $URL/configs/extract_AUMC.yaml
cd ..
chmod +x run.sh
chmod +x pre_MEDS.py
```

## Step 1: Download AUMC

Download the AUMC dataset from following the instructions on https://github.com/AmsterdamUMC/AmsterdamUMCdb?tab=readme-ov-file. You will need the raw `.csv` files for this example. We will use `$AUMC_RAW_DIR` to denote the root directory of where the resulting _core data files_ are stored.

## Step 2: Run the MEDS ETL

To run the MEDS ETL, run the following command:

```bash
./run.sh $AUMC_RAW_DIR $AUMC_PRE_MEDS_DIR $AUMC_MEDS_COHORT_DIR
```

> \[!NOTE\]
> This can take up large amounts of memory if not parallelized. You can reduce the shard size to reduce memory usage by setting the `shard_size` parameter in the `extract_AUMC.yaml` file.
> Check that your environment variables are set correctly.

To use a specific stage runner file (e.g., to set different parallelism options), you can specify it as an
additional argument

```bash
export N_WORKERS=5
./run.sh $AUMC_RAW_DIR $AUMC_PRE_MEDS_DIR $AUMC_MEDS_DIR \
    stage_runner_fp=slurm_runner.yaml
```

The `N_WORKERS` environment variable set before the command controls how many parallel workers should be used
at maximum.

The `slurm_runner.yaml` file (downloaded above) runs each stage across several workers on separate slurm
worker nodes using the `submitit` launcher. _**You will need to customize this file to your own slurm system
so that the partition names are correct before use.**_ The memory and time costs are viable in the current
configuration, but if your nodes are sufficiently different you may need to adjust those as well.

The `local_parallelism_runner.yaml` file (downloaded above) runs each stage via separate processes on the
launching machine. There are no additional arguments needed for this stage beyond the `N_WORKERS` environment
variable and there is nothing to customize in this file.

To profile the time and memory costs of your ETL, add the `do_profile=true` flag at the end.

## Notes

Note: If you use the slurm system and you launch the hydra submitit jobs from an interactive slurm node, you
may need to run `unset SLURM_CPU_BIND` in your terminal first to avoid errors.

## Future Work

Check with AUMCdb authors:

- How should we deal with `registeredat` and `updatedat`?
- We **IGNORE** several flags for the `drugitems` -- this may be a mistake!
- When is the administered dose recorded? Is this done after the fact?
