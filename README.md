# MEDS polars functions

A simple set of MEDS polars-based ETL and transformation functions. These functions include basic utilities
for putting data into MEDS format, pre-processing MEDS formatted data for deep learning modeling, tensorizing
MEDS data for efficient, at-scale deep learning training. Close analogs of these functions have been used to
successfully process data at the scale of billions of clinical events and millions of patients with modest
computational resource requirements in several hours of total runtime.

More details to come soon. In the meantime, see [this google
doc](https://docs.google.com/document/d/14NKaIPAMKC1bXWV_IVJ7nQfMo09PpUQCRrqqVY6qVT4/edit?usp=sharing) for
more information.

## Overview

This package provides three things:

1. A working, scalable, simple example of how to extract and pre-process MEDS data for downstream modeling.
   These examples are provided in the form of:
   - A set of integration tests that are run over synthetic data to verify correctness of the ETL pipeline.
     See `tests/test_extraction.py` for the ETL tests with the in-built synthetic source data.
   - A working MIMIC-IV MEDS ETL pipeline that can be run over MIMIC-IV v2.2 in approximately 1 hour in serial
     mode (and much faster if parallelized). See `MIMIC-IV_Example` for more details.
2. A flexible ETL for extracting MEDS data from a variety of source formats.
3. A pre-processing pipeline that can be used for models that require:
   - Filtering data to only include patients with a certain number of events
   - Filtering events to only include those whose codes occur a certain number of times
   - Removing numerical values that are more than a certain number of standard deviations from the mean on a
     per-code basis.
   - Normalizing numerical values on a per-code basis.
   - Modeling data in a 3D format, with patients X unique timestamps X codes & numerical values per
     timestamp.

## Installation

- For a base installation, clone this repository and run `pip install .` from the repository root.
- For running the MIMIC-IV example, install the optional MIMIC dependencies as well with `pip install .[mimic]`.
- To support same-machine, process-based parallelism, install the optional joblib dependencies with `pip install .[local_parallelism]`.
- To support cluster-based parallelism, install the optional submitit dependencies with `pip install .[slurm_parallelism]`.
- For working on development, install the optional development dependencies with `pip install .[dev,tests]`.
- Optional dependencies can be mutually installed by combining the optional dependency names with commas in
  the square brackets, e.g., `pip install .[mimic,local_parallelism]`.

## Usage -- High Level

The MEDS ETL and pre-processing pipelines are designed to be run in a modular, stage-based manner, with each
stage of the pipeline being run as a separate script. For a single pipeline, all scripts will take the same
arguments by leveraging the same Hydra configuration file, and to run multiple workers on a single stage in
parallel, the user can launch the same script multiple times _without changing the arguments or configuration
file_, and the scripts will automatically handle the parallelism and avoid duplicative work. This permits
tremendous flexibility in how these pipelines can be run.

- The user can run the entire pipeline in serial, through a single shell script simply by calling each
  stage's script in sequence.
- The user can leverage arbitrary scheduling systems (e.g., Slurm, LSF, Kubernetes, etc.) to run each stage
  in parallel on a cluster, by constructing the appropriate worker scripts to run each stage's script and
  simply launching as many worker jobs as is desired (note this will typically required a distributed file
  system to work correctly, as these scripts use manually created file locks to avoid duplicative work).
- The user can run each stage in parallel on a single machine by launching multiple copies of the same
  script in different terminal sessions. This can result in a significant speedup depending on the machine
  configuration as it ensures that parallelism can be used with minimal file read contention.

Two of these methods of parallelism, in particular local-machine parallelism and slurm-based cluster
parallelism, are supported explicitly by this package through the use of the `joblib` and `submitit` Hydra
plugins and Hydra's multirun capabilities, which will be discussed in more detail below.

By following this design convention, each individual stage of the pipeline can be kept extremely simple (often
each stage corresponds simply to a single short "dataframe" function), can be rigorously tested, can be cached
after completion to permit easy re-suming or re-running of the pipeline, and permits extremely flexible and
efficient (through parallelization) use of the pipeline in a variety of environments, all without imposing
significant complexity, overhead, or computational dependencies on the user.

Below we walk through usage of this mechanism for both the ETL and the model-specific pre-processing
pipelines in more detail.

### Scripts for the ETL Pipeline

The ETL pipeline (which is more complete, and likely to be viable for a wider range of input datasets out of
the box) relies on the following configuration files and scripts:

Configuration: `configs/extraction.yaml`

```yaml
# The event conversion configuration file is used throughout the pipeline to define the events to extract.
event_conversion_config_fp: ???

stages:
  - shard_events
  - split_and_shard_patients
  - convert_to_sharded_events
  - merge_to_MEDS_cohort

stage_configs:
  shard_events:
    row_chunksize: 200000000
    infer_schema_length: 10000
  split_and_shard_patients:
    is_metadata: true
    output_dir: ${cohort_dir}
    n_patients_per_shard: 50000
    external_splits_json_fp:
    split_fracs:
      train: 0.8
      tuning: 0.1
      held_out: 0.1
  merge_to_MEDS_cohort:
    output_dir: ${cohort_dir}/final_cohort
```

Scripts:

1. `shard_events.py`: Shards the input data into smaller, event-level shards.
2. `split_and_shard_patients.py`: Splits the patient population into ML splits and shards these splits into
   patient-level shards.
3. `convert_to_sharded_events.py`: Converts the input, event-level shards into the MEDS event format and
   sub-shards them into patient-level sub-shards.
4. `merge_to_MEDS_cohort.py`: Merges the patient-level, event-level shards into full patient-level shards.

See the `MIMIC-IV_Example` directory for a full, worked example of the ETL on MIMIC-IV v2.2.

## MEDS ETL / Extraction Pipeline Details

### Overview

Assumptions:

1. Your data is organized in a set of parquet files on disk such that each row of each file corresponds to
   one or more measurements per patient and has all necessary information in that row to extract said
   measurement, organized in a simple, columnar format. Each of these parquet files stores the patient's ID in
   a column called `patient_id` in the same type.
2. You have a pre-defined or can externally define the requisite MEDS base `code_metadata` file that
   describes the codes in your data as necessary. This file is not used in the provided pre-processing
   pipeline in this package, but is necessary for other uses of the MEDS data.

Computational Resource Requirements:

1. This pipeline is designed for achieving throughput through parallelism in a controllable and
   resource-efficient manner. By changing the input shard size and by launching more or fewer copies of the
   job steps, you can control the resources used as desired.
2. This pipeline preferentially uses disk over memory and compute through aggressive caching. You should
   have sufficient disk space to store multiple copies of your raw dataset comfortably.
3. This pipeline can be run on a single machine or across many worker nodes on a cluster provided the worker
   nodes have access to a distributed file system. The internal "locking" mechanism used to limit race
   conditions among multiple workers in this pipeline is not guaranteed to be robust to all distributed
   systems, though in practice this is unlikely to cause issues.

The provided ETL consists of the following steps, which can be performed as needed by the user with whatever
degree of parallelism is desired per step.

1. It re-shards the input data into a set of smaller, event-level shards to facilitate parallel processing.
   This can be skipped if your input data is already suitably sharded at either a per-patient or per-event
   level.
2. It extracts the subject IDs from the sharded data and computes the set of ML splits and (per split) the
   patient shards. These are stored in a JSON file in the output cohort directory.
3. It converts the input, event level shards into the MEDS flat format and joins and shards these data into
   patient-level shards for MEDS use and stores them in a nested format in the output cohort directory,
   again in the flat format. This step can be broken down into two sub-steps:
   - First, each input shard is converted to the MEDS flat format and split into sub patient-level shards.
   - Second, the appropriate sub patient-level shards are joined and re-organized into the final
     patient-level shards. This method ensures that we minimize the amount of read contention on the input
     shards during the join process and can maximize parallel throughput, as (theoretically, with sufficient
     workers) all input shards can be sub-sharded in parallel and then all output shards can be joined in
     parallel.

The ETL scripts all use [Hydra](https://hydra.cc/) for configuration management, leveraging the shared
`configs/extraction.yaml` file for configuration. The user can override any of these settings in the normal
way for Hydra configurations.

If desired, appropriate scripts can be written and run at a per-patient shard level to convert between the
flat format and any of the other valid nested MEDS format, but for now we leave that up to the user.

#### Input Event Extraction

Input events extraction configurations are defined through a simple configuration file language, stored in
YAML form on disk, which specified for a collection of events how the individual rows from the various input
dataframes should be parsed into different event formats. The YAML file stores a simple dictionary with the
following structure:

```yaml
patient_id: $GLOBAL_PATIENT_ID_OVERWRITE # Optional, if you want to overwrite the patient ID column name for
                                         # all inputs. If not specified, defaults to "patient_id".
$INPUT_FILE_STEM:
    patient_id: $INPUT_FILE_PATIENT_ID # Optional, if you want to overwrite the patient ID column name for
                                       # this input. IF not specified, defaults to the global patient ID.
    $EVENT_NAME:
        code:
          - $CODE_PART_1
          - $CODE_PART_2
          ... # These will be combined with "//" to form the final code.
        timestamp: $TIMESTAMP
        $MEDS_COLUMN_NAME: $RAW_COLUMN_NAME
        ...
    ...
...
```

In this structure, `$INPUT_FILE_STEM` is the stem of the input file name, `$EVENT_NAME` is the name of a
particular kind of event that can be extracted from the input file, `$CODE` is the code for the event, either
as a constant string or (with the syntax `"col($COLUMN)"` the name of the column in the raw data to be read to
get the code), and `$TIMESTAMP` is the timestamp for the event, either as `null` to indicate the event has a
null timestamp (e.g., a static measurement) or with the `"col($COLUMN)"` syntax refenced above, and all
subsequent key-value pairs are mappings from the MEDS column name to the raw column name in the input data.
Here, these mappings can _only_ point to columns in the input data, not constant values, and the input data
columns must be either string or categorical types (in which case they will be converted to categorical) or
numeric types. You can see this extraction logic in the `scripts/extraction/convert_to_sharded_events.py`
file, in the `extract_event` function.

### Scripts and Examples

See `tests/test_extraction.py` for an example of the end-to-end ETL pipeline being run on synthetic data. This
script is a functional test that is also run with `pytest` to verify correctness of the algorithm.

#### Core Scripts:

1. `scripts/extraction/shard_events.py` shards the input data into smaller, event-level shards by splitting
   raw files into chunks of a configurable number of rows. Files are split sequentially, with no regard for
   data content or patient boundaries. The resulting files are stored in the `subsharded_events`
   subdirectory of the output directory.
2. `scripts/extraction/split_and_shard_patients.py` splits the patient population into ML splits and shards
   these splits into patient-level shards. The result of this process is only a simple `JSON` file
   containing the patient IDs belonging to individual splits and shards. This file is stored in the
   `output_directory/splits.json` file.
3. `scripts/extraction/convert_to_sharded_events.py` converts the input, event-level shards into the MEDS
   event format and splits them into patient-level sub-shards. So, the resulting files are sharded into
   patient-level, then event-level groups and are not merged into full patient-level shards or appropriately
   sorted for downstream use.
4. `scripts/extraction/merge_to_MEDS_cohort.py` merges the patient-level, event-level shards into full
   patient-level shards and sorts them appropriately for downstream use. The resulting files are stored in
   the `output_directory/final_cohort` directory.

## MEDS Pre-processing Transformations

Once the MEDS dataset is created, in needs to be effectively pre-processed for downstream use. This package
contains a variety of pre-processing transformations and scripts that can be applied on diverse MEDS datasets
in various ways to prepare them for downstream modeling. Broadly speaking, the pre-processing pipeline can be
broken down into the following steps:

1. Filtering the dataset by criteria that do not require cross-patient analyses, e.g.,

   - Filtering patients by the number of events or unique timestamps they have.
   - Removing numerical values that fall outside of pre-specified, per-code ranges (e.g., for outlier
     removal).

2. Adding any extra events to the records that are necessary for downstream modeling, e.g.,

   - Adding time-derived measurements, e.g.,
     - The time since the last event of a certain type.
     - The patient's age as of each unique timepoint.
     - The time-of-day of each event.
     - Adding a "dummy" event to the dataset for each patient that occurs at the end of the observation
       period.

3. Iteratively (a) grouping the dataset by `code` and associated code modifier columns and collecting
   statistics on the numerical and categorical values for each code then (b) filtering the dataset down to
   remove outliers or other undesired codes or values, e.g.,

   - Computing the mean and standard deviation of the numerical values for each code.
   - Computing the number of times each code occurs in the dataset.
   - Computing appropriate numerical bins for each code for value discretization.

4. Transforming the code space to appropriately include or exclude any additional measurement columns that
   should be included during code grouping and modeling operations. The goal of this step is to ensure that
   the only columns that need be processed going into the pre-processing, tokenization, and tensorization
   stage are expressible in the `code` and `numerical_values` columns of the dataset, which helps
   standardize further downstream use.

   - Standardizing the unit of measure of observed codes or adding the unit of measure to the code such that
     group-by operations over the code take the unit into account.
   - Adding categorical normal/abnormal flags to laboratory test result codes.

5. Normalizing the data to convert codes to indices and numerical values to the desired form (either
   categorical indices or normalized numerical values).

6. Tokenizing the data in time to create a pre-tensorized dataset with clear delineations between patients,
   patient sequence elements, and measurements per sequence element (note that various of these delineations
   may be fully flat/trivial for unnested formats).

7. Tensorizing the data to permit efficient retrieval from disk of patient data for deep-learning modeling
   via PyTorch.

Much like how the entire MEDS ETL pipeline is controlled by a single configuration file, the pre-processing
pipeline is also controlled by a single configuration file, stored in `configs/preprocessing.yaml`. Scripts
leverage this file once again through the [Hydra](https://hydra.cc/) configuration management system. Similar
to the ETL, this pipeline is designed to enable seamless parallelism and efficient use of resources simply by
running multiple copies of the same script on independent workers to process the data in parallel. "Reduction"
steps again need to happen in a single-threaded manner, but these steps are generally very fast and should not
be a bottleneck.

## Overview of configuration manipulation

### Pipeline configuration: Stages and OmegaConf Resolvers

The pipeline configuration file for both the provided extraction and pre-processing pipelines are structured
to permit both ease of understanding, flexibility for user-derived modifications, and ease of use in the
simple, file-in/file-out scripts that this repository promotes. How this works is that each pipeline
(extraction and pre-processing) defines one global configuration file which is used as the Hydra specification
for all scripts in that pipeline. This file leverages some generic pipeline configuration options, specified
in `pipeline.yaml` and imported via the Hydra `defaults:` list, but also defines a list of stages with
stage-specific configurations.

The user can specify the stage in question on the command line either manually (e.g., `stage=stage_name`) or
allow the stage name to be inferred automatically from the script name. Each script receives both the global
configuration file but also a sub-configuration (within the `stage_cfg` node in the received global
configuration) which is pre-populated with the stage-specific configuration for the stage in question and
automatically inferred input and output file paths (if not overwritten in the config file) based on the stage
name and its position in the overall pipeline. This makes it easy to leverage transformations and scripts
defined here in new configuration pipelines, simply by placing them as a stage in a broader pipeline in a
different configuration or order relative to other stages.

### Running the Pipeline in Parallel via Hydra Multirun

We support two (optional) hydra multirun job launchers for parallelizing ETL and pre-processing pipeline
steps: [`joblib`](https://hydra.cc/docs/plugins/joblib_launcher/) (for local parallelism) and
[`submitit`](https://hydra.cc/docs/plugins/submitit_launcher/) to launch things with slurm for cluster
parallelism.

To use either of these, you need to install additional optional dependencies:

1. `pip install -e .[local_parallelism]` for joblib local parallelism support, or
2. `pip install -e .[slurm_parallelism]` for submitit cluster parallelism support.

## TODOs:

1. We need to have a vehicle to cleanly separate dataset-specific variables from the general configuration
   files. Similar to task configuration files, but for models.
2. Figure out how to ensure that each pre-processing step reads from the right prior files. Likely need some
   kind of a "prior stage name" config variable.
