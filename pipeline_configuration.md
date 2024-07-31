# Pipeline Configuration in MEDS-Transform

MEDS Transform's key design philosophy is that you should be able to design the necessary data pre-processing
and transformation pipeline you need simply by composing a series of simple, reusable, efficient, and
configurable local transformations into a larger pipeline. This can allow researchers to balance having highly
flexible, customizable pipelines with ensuring that all operations in the pipeline are simple, shareable,
efficient, and easy to validate.

## Stages

### File System Management

Suppose you have a pipeline with an input directory of `$INPUT_DIR` and a cohort (output) directory of
`$COHORT_DIR`. Let us further suppose we impose a series of the following stages:

1. `stage_1`: A metadata map-reduce stage (e.g., counting the occurence rates of the codes in the
   data).
2. `stage_2`: A metadata-only processing stage (e.g., filtering the code dataframe to only codes
   that occur more than 10 times).
3. `stage_3`: A data processing stage (e.g., filtering the data to only rows with a code that is in the
   current running metadata file, which, due to `stage_2`, are those codes that occur more than 10 times).
4. `stage_4`: A metadata map-reduce stage (e.g., computing the means and variances for the numerical values
   in the data).
5. `stage_5`: A data processing stage (e.g., occluding all measurement values that occur more than 3
   standard deviations from the mean).
6. `stage_6`: A metadata map-reduce stage (e.g., computing the means and variances for the numerical values
   in the data).
7. `stage_7`: A data processing stage (e.g., normalizing the data to have a mean of 0 and a standard
   deviation of 1).

Each of these stages will read and write their output datasets in the following manner.

1. `stage_1`:
   \* As there is no preceeding data stage, this stage will read the data in from `$INPUT_DIR/data`
   (the `data` suffix is the default data directory for MEDS datasets).
   \* This stage will, in its mapping stage, write the partial extracted metadata files to the
   `$COHORT_DIR/stage_1/$SHARD_NAME.parquet` directory.
   \* This stage will read in the prior joint metadata file from the `$INPUT_DIR/metadata/codes.parquet`
   directory to join with the new metadata.
   \* This stage will join all its metadata shards, join any prior columns from the old metadata, and
   write the final, joined metadata file to the `$COHORT_DIR/stage_1/codes.parquet` directory.
2. `stage_2`:
   \* This stage will read in the metadata from the `$COHORT_DIR/stage_1/codes.parquet` directory.
   \* This stage will write the filtered metadata to the `$COHORT_DIR/stage_2/codes.parquet` directory.
3. `stage_3`:
   \* This stage will read in the data from the `$INPUT_DIR/data` directory as there has still been no
   prior data processing stage. Individual shards will be read from the
   `$INPUT_DIR/data/$SHARD_NAME.parquet` files.
   \* This stage will read in the metadata from the `$COHORT_DIR/stage_2/codes.parquet` directory.
   \* This stage will write the filtered shards to the `$COHORT_DIR/stage_3/$SHARD_NAME.parquet` files.
4. `stage_4`: ... TODO
