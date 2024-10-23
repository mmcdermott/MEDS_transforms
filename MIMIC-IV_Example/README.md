# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run from this directory or from the directory in which the files in Step 0.5. were downloaded.

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
export MIMICIV_RAW_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export MIMICIV_PRE_MEDS_DIR=??? # set to the directory in which you want to store the intermediate MEDS MIMIC-IV data
export MIMICIV_MEDS_COHORT_DIR=??? # set to the directory in which you want to store the final MEDS MIMIC-IV data

export VERSION=0.0.6 # or whatever version you want
export URL="https://raw.githubusercontent.com/mmcdermott/MEDS_transforms/$VERSION/MIMIC-IV_Example"

wget $URL/run.sh
wget $URL/pre_MEDS.py
wget $URL/local_parallelism_runner.yaml
wget $URL/slurm_runner.yaml
mkdir configs
cd configs
wget $URL/configs/extract_MIMIC.yaml
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
cd $MIMICIV_RAW_DIR
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
./run.sh $MIMICIV_RAW_DIR $MIMICIV_PRE_MEDS_DIR $MIMICIV_MEDS_COHORT_DIR do_unzip=true
```
> [!NOTE] 
> This can take up large amounts of memory if not parallelized. You can reduce the shard size to reduce memory usage by setting the `shard_size` parameter in the `extract_MIMIC.yaml` file.
> Check that your environment variables are set correctly.
To not unzip the `.csv.gz` files, set `do_unzip=false` instead of `do_unzip=true`.

To use a specific stage runner file (e.g., to set different parallelism options), you can specify it as an
additional argument

```bash
export N_WORKERS=5
./run.sh $MIMICIV_RAW_DIR $MIMICIV_PRE_MEDS_DIR $MIMICIV_MEDS_DIR do_unzip=true \
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

### Pre-MEDS Processing

If you wanted, some other processing could also be done here, such as:

1. Converting the subject's dynamically recorded race into a static, most commonly recorded race field.
