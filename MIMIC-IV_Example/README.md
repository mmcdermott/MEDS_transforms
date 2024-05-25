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
- [ ] Testing the `configs/event_configs.yaml` configuration on MIMIC-IV
- [ ] Testing the steps of the MEDS extraction ETL on MIMIC-IV (this should be expected to work, but needs
  live testing).

## Step 0: Installation

Download this repository and install the requirements:

```bash
git clone git@github.com:mmcdermott/MEDS_polars_functions.git
cd MEDS_polars_functions
git checkout MIMIC_IV
conda create -n MEDS python=3.12
conda activate MEDS
pip install .[mimic]
```

## Step 1: Download MIMIC-IV

Download the MIMIC-IV dataset from https://physionet.org/content/mimiciv/2.2/ following the instructions on
that page. You will need the raw `.csv.gz` files for this example. We will use `$MIMICIV_RAW_DIR` to denote
the root directory of where the resulting _core data files_ are stored -- e.g., there should be a `hosp` and
`icu` subdirectory of `$MIMICIV_RAW_DIR`.

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
will be used as the input to the actual MEDS extraction ETL. We'll use `$MIMICIV_PREMEDS_DIR` to denote this
directory.

To run this step, you can use the following script (assumed to be run **not** from this directory but from the
root directory of this repository):

```bash
./MIMIC_IV/pre_MEDS.py raw_cohort_dir=$MIMICIV_RAW_DIR output_dir=$MIMICIV_PREMEDS_DIR
```

## Step 3: Run the MEDS extraction ETL

We will assume you want to output the final MEDS dataset into a directory we'll denote as `$MIMICIV_MEDS_DIR`.
Note this is a different directory than the pre-MEDS directory (though, of course, they can both be
subdirectories of the same root directory).

This is a step in 4 parts:

1. Sub-shard the raw files. Run this command as many times simultaneously as you would like to have workers
   performing this sub-sharding step.

```bash
./scripts/extraction/shard_events.py \
    raw_cohort_dir=$MIMICIV_PREMEDS_DIR \
    output_dir=$MIMICIV_MEDS_DIR \
    event_conversion_config_fp=./MIMIC_IV/configs/event_configs.yaml
```

2. Extract and form the patient splits and sub-shards.

```bash
./scripts/extraction/split_and_shard_patients.py \
    raw_cohort_dir=$MIMICIV_PREMEDS_DIR \
    output_dir=$MIMICIV_MEDS_DIR \
    event_conversion_config_fp=./MIMIC_IV/configs/event_configs.yaml
```

3. Extract patient sub-shards and convert to MEDS events.

```bash
./scripts/extraction/convert_to_sharded_events.py \
    raw_cohort_dir=$MIMICIV_PREMEDS_DIR \
    output_dir=$MIMICIV_MEDS_DIR \
    event_conversion_config_fp=./MIMIC_IV/configs/event_configs.yaml
```

4. Merge the MEDS events into a single file per patient sub-shard.

```bash
./scripts/extraction/merge_to_MEDS_cohort.py \
    raw_cohort_dir=$MIMICIV_PREMEDS_DIR \
    output_dir=$MIMICIV_MEDS_DIR \
    event_conversion_config_fp=./MIMIC_IV/configs/event_configs.yaml
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
