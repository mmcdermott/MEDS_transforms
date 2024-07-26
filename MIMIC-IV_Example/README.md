# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

**Status**: This is a work in progress. The code is not yet functional. Remaining work includes:

- [x] Implementing the pre-MEDS processing step.
  - [x] Implement the joining of discharge times.
  - [x] Implement the conversion of the DOB to a more usable format.
  - [x] Implement the joining of death times.
- [ ] Testing the pre-MEDS processing step on live MIMIC-IV.
  - [x] Test that it runs at all.
  - [ ] Test that the output is as expected.
- [ ] Check the installation instructions on a fresh client.
- [x] Testing the `configs/event_configs.yaml` configuration on MIMIC-IV
- [x] Testing the MEDS extraction ETL runs on MIMIC-IV (this should be expected to work, but needs
  live testing).
  - [x] Sub-sharding
  - [x] Patient split gathering
  - [x] Event extraction
  - [x] Merging
- [ ] Validating the output MEDS cohort
  - [x] Basic validation (even though file sizes are weird, the number of rows looks consistent).
  - [ ] Debug and remove rows with null codes! (there are a lot of them)
  - [ ] Detailed validation

Note: If you use the slurm system and you launch the hydra submitit jobs from an interactive slurm node, you
may need to run `unset SLURM_CPU_BIND` in your terminal first to avoid errors.

## Step 0: Installation

Download this repository and install the requirements:

```bash
git clone git@github.com:mmcdermott/MEDS_transforms.git
cd MEDS_transforms
conda create -n MEDS python=3.12
conda activate MEDS
pip install .[examples]
```

## Step 1: Download MIMIC-IV

Download the MIMIC-IV dataset from https://physionet.org/content/mimiciv/2.2/ following the instructions on
that page. You will need the raw `.csv.gz` files for this example. We will use `$MIMICIV_RAW_DIR` to denote
the root directory of where the resulting _core data files_ are stored -- e.g., there should be a `hosp` and
`icu` subdirectory of `$MIMICIV_RAW_DIR`.

## Step 1.5: Download MIMIC-IV Metadata files
```bash
cd $MIMIC_RAW_DIR
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/d_labitems_to_loinc.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/inputevents_to_rxnorm.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/lab_itemid_to_loinc.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/meas_chartevents_main.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/meas_chartevents_value.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/numerics-summary.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/outputevents_to_loinc.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/proc_datetimeevents.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/proc_itemid.csv
wget https://raw.githubusercontent.com/MIT-LCP/mimic-code/v2.4.0/mimic-iv/concepts/concept_map/waveforms-summary.csv
```

## Step 2: Run the basic MEDS ETL

This step contains several sub-steps; luckily, all these substeps can be run via a single script, with the
`joint_script.sh` script which uses the Hydra `joblib` launcher to run things with local parallelism (make
sure you enable this feature by including the `[local_parallelism]` option during installation) or via
`joint_script_slurm.sh` which uses the Hydra `submitit` launcher to run things through slurm (make sure you
enable this feature by including the `[slurm_parallelism]` option during installation). This script entails
several steps:

### Step 2.1: Get the data ready for base MEDS extraction

This is a step in a few parts:

1. Join a few tables by `hadm_id` to get the right timestamps in the right rows for processing. In
   particular, we need to join:
   - the `hosp/diagnoses_icd` table with the `hosp/admissions` table to get the `dischtime` for each
     `hadm_id`.
   - the `hosp/drgcodes` table with the `hosp/admissions` table to get the `dischtime` for each `hadm_id`.
2. Convert the patient's static data to a more parseable form. This entails:
   - Get the patient's DOB in a format that is usable for MEDS, rather than the integral `anchor_year` and
     `anchor_offset` fields.
   - Merge the patient's `dod` with the `deathtime` from the `admissions` table.

After these steps, modified files or symlinks to the original files will be written in a new directory which
will be used as the input to the actual MEDS extraction ETL. We'll use `$MIMICIV_PREMEDS_DIR` to denote this
directory.

This step is run in the `joint_script.sh` script or the `joint_script_slurm.sh` script, but in either case the
base command that is run is as follows (assumed to be run **not** from this directory but from the
root directory of this repository):

```bash
./MIMIC-IV_Example/pre_MEDS.py raw_cohort_dir=$MIMICIV_RAW_DIR output_dir=$MIMICIV_PREMEDS_DIR
```

In practice, on a machine with 150 GB of RAM and 10 cores, this step takes less than 5 minutes in total.

### Step 2.2: Run the MEDS extraction ETL

We will assume you want to output the final MEDS dataset into a directory we'll denote as `$MIMICIV_MEDS_DIR`.
Note this is a different directory than the pre-MEDS directory (though, of course, they can both be
subdirectories of the same root directory).

This is a step in 4 parts:

1. Sub-shard the raw files. Run this command as many times simultaneously as you would like to have workers
   performing this sub-sharding step. See below for how to automate this parallelism using hydra launchers.

   This step uses the `./scripts/extraction/shard_events.py` script. See `joint_script*.sh` for the expected
   format of the command.

2. Extract and form the patient splits and sub-shards. The `./scripts/extraction/split_and_shard_patients.py`
   script is used for this step. See `joint_script*.sh` for the expected format of the command.

3. Extract patient sub-shards and convert to MEDS events. The
   `./scripts/extraction/convert_to_sharded_events.py` script is used for this step. See `joint_script*.sh` for
   the expected format of the command.

4. Merge the MEDS events into a single file per patient sub-shard. The
   `./scripts/extraction/merge_to_MEDS_cohort.py` script is used for this step. See `joint_script*.sh` for the
   expected format of the command.

5. (Optional) Generate preliminary code statistics and merge to external metadata. This is not performed
   currently in the `joint_script*.sh` scripts.

## Pre-processing for a model

To run the pre-processing steps for a model, consider the sample script provided here:

1. Filter patients to only those with at least 32 events (unique timepoints):

```bash
mbm47 in  compute-a-17-72 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines
❯ ./scripts/preprocessing/filter_patients.py --multirun worker="range(0,3)" hydra/launcher=joblib input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DIR/test" code_modifier_columns=null stage_configs.filter_patients.min_events_per_patient=32
```

2. Add time-derived measurements (age and time-of-day):

```bash
mbm47 in  compute-a-17-72 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines took 3s
❯ ./scripts/preprocessing/add_time_derived_measurements.py --multirun worker="range(0,3)" hydra/launcher=joblib input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DI
R/test" code_modifier_columns=null stage_configs.add_time_derived_measurements.age.DOB_code="DOB"
```

3. Get preliminary counts for code filtering:

```bash
mbm47 in  compute-a-17-72 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines
❯ ./scripts/preprocessing/collect_code_metadata.py --multirun worker="range(0,3)" hydra/launcher=joblib input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DIR/test" code_modifier_columns=null stage="preliminary_counts"
```

4. Filter codes:

```bash
mbm47 in  compute-a-17-72 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines took 4s
❯ ./scripts/preprocessing/filter_codes.py --multirun worker="range(0,3)" hydra/launcher=joblib input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DIR/test" code_modi
fier_columns=null stage_configs.filter_codes.min_patients_per_code=128 stage_configs.filter_codes.min_occurrences_per_code=256
```

5. Get outlier detection params:

```bash
mbm47 in  compute-a-17-72 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines took 19m57s
❯ ./scripts/preprocessing/collect_code_metadata.py --multirun worker="range(0,3)" hydra/launcher=joblib input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DIR/test" code_modifier_columns=null stage=fit_outlier_detection
```

6. Filter outliers:

```bash
mbm47 in  compute-a-17-72 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines took 5m14s
❯ ./scripts/preprocessing/filter_outliers.py --multirun worker="range(0,3)" hydra/launcher=joblib input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DIR/test" code_modifier_columns=null
```

7. Fit normalization parameters:

```bash
mbm47 in  compute-a-17-72 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines took 16m25s
❯ ./scripts/preprocessing/collect_code_metadata.py --multirun worker="range(0,3)" hydra/launcher=joblib input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DIR/test" code_modifier_columns=null stage=fit_normalization
```

8. Fit vocabulary:

```bash
mbm47 in  compute-e-16-230 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines took 2s
❯ ./scripts/preprocessing/fit_vocabulary_indices.py input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DIR/test" code_modifier_columns=null
```

9. Normalize:

```bash
mbm47 in  compute-e-16-230 in MEDS_transforms on  preprocessing_steps [$] is 󰏗 v0.0.1 via  v3.12.3 via  MEDS_pipelines took 4s
❯ ./scripts/preprocessing/normalize.py --multirun worker="range(0,3)" hydra/launcher=joblib input_dir="$MIMICIV_MEDS_DIR/3workers_slurm" cohort_dir="$MIMICIV_MEDS_PROC_DIR/test" code_modifie
r_columns=null
```

## Limitations / TO-DOs:

Currently, some tables are ignored, including:

1. `hosp/emar_detail`
2. `hosp/microbiologyevents`
3. `hosp/services`
4. `icu/datetimeevents`
5. `icu/ingredientevents`

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
