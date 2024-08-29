# eICU-CRD Example

## TODO -- This is not yet adapted to the 0.0.1 interface, but will be incorporated soon!

This is an example of how to extract a MEDS dataset from [eICU-CRD
v2.0](https://physionet.org/content/eicu-crd/2.0/). All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

**Status**: This is a work in progress. The code is not yet functional. Remaining work includes:

- [ ] Implementing the pre-MEDS processing step.
    - [ ] Identifying the pre-MEDS steps for eICU
- [ ] Testing the pre-MEDS processing step on live eICU-CRD.
    - [ ] Test that it runs at all.
    - [ ] Test that the output is as expected.
- [ ] Check the installation instructions on a fresh client.
- [ ] Testing the `configs/event_configs.yaml` configuration on eICU-CRD
- [ ] Testing the MEDS extraction ETL runs on eICU-CRD (this should be expected to work, but needs
    live testing).
    - [ ] Sub-sharding
    - [ ] Patient split gathering
    - [ ] Event extraction
    - [ ] Merging
- [ ] Validating the output MEDS cohort
    - [ ] Basic validation
    - [ ] Detailed validation

## Step 0: Installation

Install the requirements and source the requisite scripts

```bash
conda create -n MEDS python=3.12
conda activate MEDS
pip install "MEDS_transforms[local_parallelism]"
mkdir eICU_Example
cd eICU_Example
wget https://raw.githubusercontent.com/mmcdermott/MEDS_transforms/main/eICU_Example/joint_script.sh
wget https://raw.githubusercontent.com/mmcdermott/MEDS_transforms/main/eICU_Example/pre_MEDS.py
chmod +x joint_script.sh
chmod +x joint_script_slurm.sh
chmod +x pre_MEDS.py
cd ..
```

## Step 1: Download eICU

Download the eICU-CRD dataset (version 2.0) from https://physionet.org/content/eicu-crd/2.0/ following the
instructions on that page. You will need the raw `.csv.gz` files for this example. We will use
`$EICU_RAW_DIR` to denote the root directory of where the resulting _core data files_ are stored -- e.g.,
there should be a `hosp` and `icu` subdirectory of `$EICU_RAW_DIR`.

## Step 2: Get the data ready for base MEDS extraction

This is a step in a few parts:

1. Join a few tables by `hadm_id` to get the right timestamps in the right rows for processing. In
    particular, we need to join:
    - TODO
2. Convert the patient's static data to a more parseable form. This entails:
    - Get the patient's DOB in a format that is usable for MEDS, rather than the integral `anchor_year` and
        `anchor_offset` fields.
    - Merge the patient's `dod` with the `deathtime` from the `admissions` table.

After these steps, modified files or symlinks to the original files will be written in a new directory which
will be used as the input to the actual MEDS extraction ETL. We'll use `$EICU_PREMEDS_DIR` to denote this
directory.

To run this step, you can use the following script (assumed to be run **not** from this directory but from the
root directory of this repository):

```bash
./eICU_Example/pre_MEDS.py raw_cohort_dir=$EICU_RAW_DIR output_dir=$EICU_PREMEDS_DIR
```

In practice, on a machine with 150 GB of RAM and 10 cores, this step takes less than 5 minutes in total.

## Step 3: Run the MEDS extraction ETL

Note that eICU has a lot more observations per patient than does MIMIC-IV, so to keep to a reasonable memory
burden (e.g., \< 150GB per worker), you will want a smaller shard size, as well as to turn off the final unique
check (which should not be necessary given the structure of eICU and is expensive) in the merge stage. You can
do this by setting the following parameters at the end of the mandatory args when running this script:

- `stage_configs.split_and_shard_patients.n_patients_per_shard=10000`
- `stage_configs.merge_to_MEDS_cohort.unique_by=null`

### Running locally, serially

We will assume you want to output the final MEDS dataset into a directory we'll denote as `$EICU_MEDS_DIR`.
Note this is a different directory than the pre-MEDS directory (though, of course, they can both be
subdirectories of the same root directory).

This is a step in 4 parts:

1. Sub-shard the raw files. Run this command as many times simultaneously as you would like to have workers
    performing this sub-sharding step. See below for how to automate this parallelism using hydra launchers.

```bash
./scripts/extraction/shard_events.py \
    input_dir=$EICU_PREMEDS_DIR \
    cohort_dir=$EICU_MEDS_DIR \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml
```

In practice, on a machine with 150 GB of RAM and 10 cores, this step takes approximately 20 minutes in total.

1. Extract and form the patient splits and sub-shards.

```bash
./scripts/extraction/split_and_shard_patients.py \
    input_dir=$EICU_PREMEDS_DIR \
    cohort_dir=$EICU_MEDS_DIR \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml
```

In practice, on a machine with 150 GB of RAM and 10 cores, this step takes less than 5 minutes in total.

1. Extract patient sub-shards and convert to MEDS events.

```bash
./scripts/extraction/convert_to_sharded_events.py \
    input_dir=$EICU_PREMEDS_DIR \
    cohort_dir=$EICU_MEDS_DIR \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml
```

In practice, serially, this also takes around 20 minutes or more. However, it can be trivially parallelized to
cut the time down by a factor of the number of workers processing the data by simply running the command
multiple times (though this will, of course, consume more resources). If your filesystem is distributed, these
commands can also be launched as separate slurm jobs, for example. For eICU, this level of parallelization
and performance is not necessary; however, for larger datasets, it can be.

1. Merge the MEDS events into a single file per patient sub-shard.

```bash
./scripts/extraction/merge_to_MEDS_cohort.py \
    input_dir=$EICU_PREMEDS_DIR \
    cohort_dir=$EICU_MEDS_DIR \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml
```

### Running Locally, in Parallel.

This step is the exact same commands as above, but leverages Hydra's multirun capabilities with the `joblib`
launcher. Install this package with the optional `local_parallelism` option (e.g., `pip install -e .[local_parallelism]` and run `./eICU_Example/joint_script.sh`. See that script for expected args.

### Running Each Step over Slurm

To use slurm, run each command with the number of workers desired using Hydra's multirun capabilities with the
`submitit_slurm` launcher. Install this package with the optional `slurm_parallelism` option. See below for
modified commands. Note these can't be chained in a single script as the jobs will not wait for all slurm jobs
to finish before moving on to the next stage. Let `$N_PARALLEL_WORKERS` be the number of desired workers

1. Sub-shard the raw files.

```bash
./scripts/extraction/shard_events.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=60 \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.mem_gb=50 \
    hydra.launcher.name="${hydra.job.name}_${worker}" \
    hydra.launcher.partition="short" \
    input_dir=$EICU_PREMEDS_DIR \
    cohort_dir=$EICU_MEDS_DIR \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml
```

In practice, on a machine with 150 GB of RAM and 10 cores, this step takes approximately 20 minutes in total.

1. Extract and form the patient splits and sub-shards.

```bash
./scripts/extraction/split_and_shard_patients.py \
    input_dir=$EICU_PREMEDS_DIR \
    cohort_dir=$EICU_MEDS_DIR \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml
```

In practice, on a machine with 150 GB of RAM and 10 cores, this step takes less than 5 minutes in total.

1. Extract patient sub-shards and convert to MEDS events.

```bash
./scripts/extraction/convert_to_sharded_events.py \
    input_dir=$EICU_PREMEDS_DIR \
    cohort_dir=$EICU_MEDS_DIR \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml
```

In practice, serially, this also takes around 20 minutes or more. However, it can be trivially parallelized to
cut the time down by a factor of the number of workers processing the data by simply running the command
multiple times (though this will, of course, consume more resources). If your filesystem is distributed, these
commands can also be launched as separate slurm jobs, for example. For eICU, this level of parallelization
and performance is not necessary; however, for larger datasets, it can be.

1. Merge the MEDS events into a single file per patient sub-shard.

```bash
./scripts/extraction/merge_to_MEDS_cohort.py \
    input_dir=$EICU_PREMEDS_DIR \
    cohort_dir=$EICU_MEDS_DIR \
    event_conversion_config_fp=./eICU_Example/configs/event_configs.yaml
```

## Limitations / TO-DOs:

Currently, some tables are ignored, including:

1. `admissiondrug`: The [documentation](https://eicu-crd.mit.edu/eicutables/admissiondrug/) notes that this is
    extremely infrequently used, so we skip it.
2.

Lots of questions remain about how to appropriately handle timestamps of the data -- e.g., things like HCPCS
events are stored at the level of the _date_, not the _datetime_. How should those be slotted into the
timeline which is otherwise stored at the _datetime_ resolution?

Other questions:

1. How to handle merging the deathtimes between the hosp table and the patients table?
2. How to handle the dob nonsense MIMIC has?

## Future Work

### Pre-MEDS Processing

If you wanted, some other processing could also be done here, such as:

1. Converting the patient's dynamically recorded race into a static, most commonly recorded race field.
