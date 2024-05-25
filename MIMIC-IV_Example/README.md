# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV.

**Status**: This is a work in progress. The code is not yet functional.

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

If you wanted, some other processing could also be done here, such as:

1. Converting the patient's dynamically recorded race into a static, most commonly recorded race field.

## Step 3: Run the MEDS extraction ETL

This is a step in 4 parts:

1. Sub-shard the raw files.
2. Extract and form the patient splits and sub-shards.
3. Extract patient sub-shards and convert to MEDS events.
4. Merge the MEDS events into a single file per patient sub-shard.

### Step 3.1: Sub-shard the raw files

Run as many copies of the following shell script as you would like to have workers in parallel performing this
sub-sharding step.

```bash
```

### Step 3.2: Extract and form the patient splits and sub-shards.

```bash
```

### Step 3.3: Extract patient sub-shards and convert to MEDS events.

```bash
```

### Step 3.4: Merge the MEDS events into a single file per patient sub-shard.

```bash
```
