# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

## Step 0: Installation

```bash
conda create -n meds-transform python=3.12
conda activate meds-transform
# Get the latest version of MEDS_transforms from pypi
LATEST_VERSION=$(pip index versions "meds-transforms" 2>/dev/null | egrep -o '([0-9]+\.){2}[0-9]+' | head -n 1)
export VERSION=$LATEST_VERSION # or whatever version you want, at the time of writing this is "0.0.8"
pip install "MEDS_transforms[local_parallelism,slurm_parallelism]==${VERSION}"
```

If you want to profile the time and memory costs of your ETL, also install: `pip install hydra-profiler`.

## Step 0.5: Set-up
Set some environment variables and download the necessary files:
```bash
export MIMICIV_RAW_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_PRE_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_MEDS_COHORT_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data

export URL="https://raw.githubusercontent.com/mmcdermott/MEDS_transforms/$VERSION/MIMIC-IV_Example"

wget $URL/run.sh
wget $URL/pre_MEDS.py
wget $URL/local_parallelism_runner.yaml
wget $URL/slurm_runner.yaml
mkdir configs
cd configs
wget $URL/configs/extract_MIMIC.yaml
wget $URL/configs/pre_MEDS.yaml
wget $URL/configs/event_configs.yaml
cd ..
chmod +x run.sh
chmod +x pre_MEDS.py
```

## Step 1: Download MIMIC-IV

Download the MIMIC-IV dataset from https://physionet.org/content/mimiciv/2.2/ following the instructions on
that page. You will need the raw `.csv.gz` files for this example. We will use `$MIMICIV_RAW_DIR` to denote
the root directory of where the resulting _core data files_ are stored -- e.g., there should be a `hosp` and
`icu` subdirectory of `$MIMICIV_RAW_DIR`.

## Step 1.5: Download MIMIC-IV Metadata files

```bash
cd $MIMIC_RAW_DIR
export MIMIC_URL=https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map
wget $MIMIC_URL/d_labitems_to_loinc.csv
wget $MIMIC_URL/inputevents_to_rxnorm.csv
wget $MIMIC_URL/lab_itemid_to_loinc.csv
wget $MIMIC_URL/meas_chartevents_main.csv
wget $MIMIC_URL/meas_chartevents_value.csv
wget $MIMIC_URL/numerics-summary.csv
wget $MIMIC_URL/outputevents_to_loinc.csv
wget $MIMIC_URL/proc_datetimeevents.csv
wget $MIMIC_URL/proc_itemid.csv
wget $MIMIC_URL/waveforms-summary.csv
```

## Step 2: Run the MEDS ETL

To run the MEDS ETL, run the following command:

```bash
./run.sh $MIMICIV_RAW_DIR $MIMICIV_PRE_MEDS_DIR $MIMICIV_MEDS_DIR do_unzip=true
```

To not unzip the `.csv.gz` files, set `do_unzip=false` instead of `do_unzip=true`.

To use a specific stage runner file (e.g., to set different parallelism options), you can specify it as an
additional argument

```bash
export N_WORKERS=8
./run.sh $MIMICIV_RAW_DIR $MIMICIV_PRE_MEDS_DIR $MIMICIV_MEDS_DIR do_unzip=false \
    stage_runner_fp=local_parallelism_runner.yaml
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

### Pre-MEDS Processing

If you wanted, some other processing could also be done here, such as:

1. Converting the subject's dynamically recorded race into a static, most commonly recorded race field.
