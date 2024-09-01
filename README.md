# MEDS Transforms

[![PyPI - Version](https://img.shields.io/pypi/v/MEDS-transforms)](https://pypi.org/project/MEDS-transforms/)
![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/meds-transforms/badge/?version=latest)](https://meds-transforms.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/mmcdermott/MEDS_transforms/graph/badge.svg?token=5RORKQOZF9)](https://codecov.io/gh/mmcdermott/MEDS_transforms)
[![tests](https://github.com/mmcdermott/MEDS_transforms/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_transforms/actions/workflows/tests.yml)
[![code-quality](https://github.com/mmcdermott/MEDS_transforms/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/MEDS_transforms/actions/workflows/code-quality-main.yaml)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/MEDS_transforms#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/MEDS_transforms/pulls)
[![contributors](https://img.shields.io/github/contributors/mmcdermott/MEDS_transforms.svg)](https://github.com/mmcdermott/MEDS_transforms/graphs/contributors)

This repository contains a set of functions and scripts for extraction to and transformation/pre-processing of
MEDS-formatted data.

Completed functions include scripts and utilities for extraction of various forms of raw data into the MEDS
format in a scalable, parallelizable manner, as well as general configuration management utilities for complex
pipelines over MEDS data. In progress functions include more model-specific pre-processing steps for MEDS
data.

Examples of these capabilities in action can be seen in the `MIMIC-IV_Example` directory,
which contains a working, end-to-end examples to extract MEDS formatted data from MIMIC-IV v2.2. A working
example for eICU v2.0 is also present, though needs to be adapted to recent interface improvements. These
directories also have `README.md` files with more detailed information on how to run the scripts in those
directories.

## Installation

- For a pypi installation, install with `pip install MEDS-transforms`.
- For a local installation, clone this repository and run `pip install .` from the repository root.
- For running the MIMIC-IV example, install the optional MIMIC dependencies as well with
    `pip install MEDS-transforms[examples]`.
- To support same-machine, process-based parallelism, install the optional joblib dependencies with
    `pip install MEDS-transforms[local_parallelism]`.
- To support cluster-based parallelism, install the optional submitit dependencies with
    `pip install MEDS-transforms[slurm_parallelism]`.
- For working on development, install the optional development dependencies with
    `pip install .[dev,tests]`.
- Optional dependencies can be mutually installed by combining the optional dependency names with commas in
    the square brackets, e.g., `pip install MEDS-transforms[examples,local_parallelism]`.

## Design Philosophy

The fundamental design philosophy of this repository can be summarized as follows:

1. _(The MEDS Assumption)_: All structured electronic health record (EHR) data can be represented as a
    series of events, each of which is associated with a subject, a time, and a set of codes and
    numeric values. This representation is the Medical Event Data Standard (MEDS) format, and in this
    repository we use it in the "flat" format, where data is organized as rows of `subject_id`,
    `time`, `code`, `numeric_value` columns.
2. _Easy Efficiency through Sharding_: MEDS datasets in this repository are sharded into smaller, more
    manageable pieces (organized as separate files) at the subject level (and, during the raw-data extraction
    process, the event level). This enables users to scale up their processing capabilities ad nauseum by
    leveraging more workers to process these shards in parallel. This parallelization is seamlessly enabled
    with the configuration schema used in the scripts in this repository. This style of parallelization
    does not require complex packages to manage, complex systems of parallelization support, and can be
    employed on single machines or across clusters. Through this style of parallelism, the MIMIC-IV ETL
    included in this repository has been run end to end in under ten minutes with suitable parallelization.
3. _Simple, Modular, and Testable_: Each stage of the pipelines demonstrated in this repository is designed
    to be simple, modular, and testable. Each operation is a single script that can be run independently of
    the others, and each stage is designed to do a small amount of work and be easily testable in isolation.
    This design philosophy ensures that the pipeline is robust to changes, easy to debug, and easy to extend.
    In particular, to add new operations specific to a given model or dataset, the user need only write
    simple functions that take in a flat MEDS dataframe (representing a single subject level shard) and
    return a new flat MEDS dataframe, and then wrap that function in a script by following the examples
    provided in this repository. These individual functions can use the same configuration schema as other
    stages in the pipeline or include a separate, stage-specific configuration, and can use whatever
    dataframe or data-processing tool desired (e.g., Pandas, Polars, DuckDB, FEMR, etc.), though the examples
    in this repository leverage Polars.
4. _Configuration Extensibility through Hydra_: We rely heavily on Hydra and OmegaConf in this repository to
    simplify configuration management within and across stages for a single pipeline. This design enables
    easy choice of parallelization by leveraging distinct Hydra launcher plugins for local or cluster-driven
    parallelism, natural capturing of logs and outputs for each stage, easy incorporation of documentation
    and help-text for both overall pipelines and individual stages, and extensibility beyond default patterns
    for more complex use-cases.
5. _Configuration Files over Code_: Wherever _sensible_, we prefer to rely on configuration files rather
    than code to specify repeated behavior prototypes over customized, dataset-specific code to enable
    maximal reproducibility. The core strength of MEDS is that it is a shared, standardized format for EHR
    data, and this repository is designed to leverage that strength to the fullest extent possible by
    designing pipelines that can be, wherever possible, run identically save for configuration file inputs
    across disparate datasets. Configuration files also can be easier to communicate to local data experts,
    who may not have Python expertise, providing another benefit. This design philosophy is not absolute,
    however, and local code can and _should_ be used where appropriate -- see the `MIMIC-IV_Example` and
    `eICU_Example` directories for examples of how and where per-dataset code can be leveraged in concert
    with the configurable aspects of the standardized MEDS extraction pipeline.

## Intended Usage

This pipeline is intended to be used both as a total or partial standalone ETL pipeline for converting raw EHR
data into the MEDS format (this operation is often much more standardized than model-specific pre-processing
needs) and as a template for model-specific pre-processing pipelines.

### Existing Scripts

The MEDS ETL and pre-processing pipelines are designed to be run in a modular, stage-based manner, with each
stage of the pipeline being run as a separate script. For a single pipeline, all scripts will take the same
arguments by leveraging the same Hydra configuration file, and to run multiple workers on a single stage in
parallel, the user can launch the same script multiple times _without changing the arguments or configuration
file_ or (to facilitate multirun capabilities) by simply changing the `worker` configuration value (this
configuration value is not used by anything except log file names), and the scripts will automatically handle
the parallelism and avoid duplicative work. This permits significant flexibility in how these pipelines can be
run.

- The user can run the entire pipeline in serial, through a single shell script simply by calling each
    stage's script in sequence.
- The user can leverage arbitrary scheduling systems (e.g., Slurm, LSF, Kubernetes, etc.) to run each stage
    in parallel on a cluster, either by manually constructing the appropriate worker scripts to run each stage's
    script and simply launching as many worker jobs as is desired or by using Hydra launchers such as the
    `submitit` launcher to automate the creation of appropriate scheduler worker jobs. Note this will typically
    required a distributed file system to work correctly, as these scripts use manually created file locks to
    avoid duplicative work.
- The user can run each stage in parallel on a single machine by launching multiple copies of the same
    script in different terminal sessions or all at once via the Hydra `joblib` launcher. This can result in a
    significant speedup depending on the machine configuration as sharding ensures that parallelism can be used
    with minimal file read contention.

Two of these methods of parallelism, in particular local-machine parallelism and slurm-based cluster
parallelism, are supported explicitly by this package through the use of the `joblib` and `submitit` Hydra
plugins and Hydra's multirun capabilities, which will be discussed in more detail below.

By following this design convention, each individual stage of the pipeline can be kept extremely simple (often
each stage corresponds to a single short "dataframe" function), can be rigorously tested, can be cached
after completion to permit easy re-suming or re-running of the pipeline, and permits extremely flexible and
efficient (through parallelization) use of the pipeline in a variety of environments, all without imposing
significant complexity, overhead, or computational dependencies on the user.

To see each of the scripts for the various pipelines, examine the `scripts` directory. Each script will, when
run with the `--help` flag, provide a detailed description of the script's purpose, arguments, and usage.
E.g.,

```bash
❯  MEDS_extract-shard_events --help
== MEDS/shard_events ==
MEDS/shard_events is a command line tool that provides an interface for running MEDS pipelines.

**Pipeline description:**
This pipeline extracts raw MEDS events in longitudinal, sparse form from an input dataset meeting select
criteria and converts them to the flattened, MEDS format. It can be run in its entirety, with controllable
levels of parallelism, or in stages. Arguments:
  - `event_conversion_config_fp`: The path to the event conversion configuration file. This file defines
    the events to extract from the various rows of the various input files encountered in the global input
    directory.
  - `input_dir`: The path to the directory containing the raw input files.
  - `cohort_dir`: The path to the directory where the output cohort will be written. It will be written in
    various subfolders of this dir depending on the stage, as intermediate stages cache their output during
    computation for efficiency of re-running and distributing.

**Stage description:**
This stage shards the raw input events into smaller files for easier processing. Arguments:
  - `row_chunksize`: The number of rows to read in at a time.
  - `infer_schema_length`: The number of rows to read in to infer the schema (only used if the source
    files are csvs)
```

Note that these stage scripts can be used for either a full pipeline or just as a component of a larger,
user-defined process -- it is up to the user to decide how to leverage these scripts in their own work.

### As an importable library

To use this repository as an importable library, the user should follow these steps:

1. Install the repository as a package.
2. Design your own transform function in your own codebase and leverage `MEDS_transform` utilities such as
    `MEDS_transform.mapreduce.mapper.map_over` to easily apply your transform over a sharded MEDS dataset.
3. Leverage the `MEDS_transforms` configuration schema to enable easy configuration of your pipeline, by
    importing the MEDS transforms configs via your hydra search path and using them as a base for your own
    configuration files, enabling you to intermix your new stage configuration with the existing MEDS
    transform stages.
4. Note that, if your transformations are sufficiently general, you can also submit a PR to add new
    transformations to this repository, enabling others to leverage your work as well.
    See [this example](https://github.com/mmcdermott/MEDS_transforms/pull/48) for an (in progress) example of
    how to do this.

### As a template

To use this repository as a template, the user should follow these steps:

1. Fork the repository to a new repository for their dedicated pipeline.
2. Design the set of "stages" (e.g., distinct operations that must be completed) that will be required for
    their needs. As a best practice, each stage should be realized as a single or set of simple functions
    that can be applied on a per-shard basis to the data. Reduction stages (where data needs to be aggregated
    across the entire pipeline) should be kept as simple as possible to avoid bottlenecks, but are supported
    through this pipeline design; see the (in progress) `scripts/preprocessing/collect_code_metadata.py`
    script for an example.
3. Mimic the structure of the `configs/preprocessing.yaml` configuration file to assemble a configuration
    file for the necessary stages of your pipeline. Identify in advance what dataset-specific information the
    user will need to specify to run your pipeline (e.g., will they need links between dataset codes and
    external ontologies? Will they need to specify select key-event concepts to identify in the data? etc.).
    Proper pipeline design should enable running the pipeline across multiple datasets with minimal
    dataset-specific information required, and such that that information can be specified in as easy a
    manner as possible. Examples of how to do this are forthcoming.

## MEDS ETL / Extraction Pipeline Details

### Overview

Assumptions:

1. Your data is organized in a set of parquet files on disk such that each row of each file corresponds to
    one or more measurements per subject and has all necessary information in that row to extract said
    measurement, organized in a simple, columnar format. Each of these parquet files stores the subject's ID in
    a column called `subject_id` in the same type.
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
    This can be skipped if your input data is already suitably sharded at either a per-subject or per-event
    level.
2. It extracts the subject IDs from the sharded data and computes the set of ML splits and (per split) the
    subject shards. These are stored in a JSON file in the output cohort directory.
3. It converts the input, event level shards into the MEDS flat format and joins and shards these data into
    subject-level shards for MEDS use and stores them in a nested format in the output cohort directory,
    again in the flat format. This step can be broken down into two sub-steps:
    - First, each input shard is converted to the MEDS flat format and split into sub subject-level shards.
    - Second, the appropriate sub subject-level shards are joined and re-organized into the final
        subject-level shards. This method ensures that we minimize the amount of read contention on the input
        shards during the join process and can maximize parallel throughput, as (theoretically, with sufficient
        workers) all input shards can be sub-sharded in parallel and then all output shards can be joined in
        parallel.

The ETL scripts all use [Hydra](https://hydra.cc/) for configuration management, leveraging the shared
`configs/extraction.yaml` file for configuration. The user can override any of these settings in the normal
way for Hydra configurations.

If desired, appropriate scripts can be written and run at a per-subject shard level to convert between the
flat format and any of the other valid nested MEDS format, but for now we leave that up to the user.

#### Input Event Extraction

Input events extraction configurations are defined through a simple configuration file language, stored in
YAML form on disk, which specified for a collection of events how the individual rows from the various input
dataframes should be parsed into different event formats. The YAML file stores a simple dictionary with the
following structure:

```yaml
subject_id: $GLOBAL_SUBJECT_ID_OVERWRITE # Optional, if you want to overwrite the subject ID column name for
                                         # all inputs. If not specified, defaults to "subject_id".
$INPUT_FILE_STEM:
    subject_id: $INPUT_FILE_SUBJECT_ID # Optional, if you want to overwrite the subject ID column name for
                                       # this input. IF not specified, defaults to the global subject ID.
    $EVENT_NAME:
        code:
          - $CODE_PART_1
          - $CODE_PART_2
          ... # These will be combined with "//" to form the final code.
        time: $TIME
        $MEDS_COLUMN_NAME: $RAW_COLUMN_NAME
        ...
    ...
...
```

In this structure, `$INPUT_FILE_STEM` is the stem of the input file name, `$EVENT_NAME` is the name of a
particular kind of event that can be extracted from the input file, `$CODE` is the code for the event, either
as a constant string or (with the syntax `"col($COLUMN)"` the name of the column in the raw data to be read to
get the code), and `$TIME` is the time for the event, either as `null` to indicate the event has a
null time (e.g., a static measurement) or with the `"col($COLUMN)"` syntax refenced above, and all
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
    data content or subject boundaries. The resulting files are stored in the `subsharded_events`
    subdirectory of the output directory.
2. `scripts/extraction/split_and_shard_subjects.py` splits the subject population into ML splits and shards
    these splits into subject-level shards. The result of this process is only a simple `JSON` file
    containing the subject IDs belonging to individual splits and shards. This file is stored in the
    `output_directory/splits.json` file.
3. `scripts/extraction/convert_to_sharded_events.py` converts the input, event-level shards into the MEDS
    event format and splits them into subject-level sub-shards. So, the resulting files are sharded into
    subject-level, then event-level groups and are not merged into full subject-level shards or appropriately
    sorted for downstream use.
4. `scripts/extraction/merge_to_MEDS_cohort.py` merges the subject-level, event-level shards into full
    subject-level shards and sorts them appropriately for downstream use. The resulting files are stored in
    the `output_directory/final_cohort` directory.

## MEDS Pre-processing Transformations

Once the MEDS dataset is created, in needs to be effectively pre-processed for downstream use. This package
contains a variety of pre-processing transformations and scripts that can be applied on diverse MEDS datasets
in various ways to prepare them for downstream modeling. Broadly speaking, the pre-processing pipeline can be
broken down into the following steps:

1. Filtering the dataset by criteria that do not require cross-subject analyses, e.g.,

    - Filtering subjects by the number of events or unique times they have.
    - Removing numeric values that fall outside of pre-specified, per-code ranges (e.g., for outlier
        removal).

2. Adding any extra events to the records that are necessary for downstream modeling, e.g.,

    - Adding time-derived measurements, e.g.,
        - The time since the last event of a certain type.
        - The subject's age as of each unique timepoint.
        - The time-of-day of each event.
        - Adding a "dummy" event to the dataset for each subject that occurs at the end of the observation
            period.

3. Iteratively (a) grouping the dataset by `code` and associated code modifier columns and collecting
    statistics on the numeric and categorical values for each code then (b) filtering the dataset down to
    remove outliers or other undesired codes or values, e.g.,

    - Computing the mean and standard deviation of the numeric values for each code.
    - Computing the number of times each code occurs in the dataset.
    - Computing appropriate numeric bins for each code for value discretization.

4. Transforming the code space to appropriately include or exclude any additional measurement columns that
    should be included during code grouping and modeling operations. The goal of this step is to ensure that
    the only columns that need be processed going into the pre-processing, tokenization, and tensorization
    stage are expressible in the `code` and `numeric_values` columns of the dataset, which helps
    standardize further downstream use.

    - Standardizing the unit of measure of observed codes or adding the unit of measure to the code such that
        group-by operations over the code take the unit into account.
    - Adding categorical normal/abnormal flags to laboratory test result codes.

5. Normalizing the data to convert codes to indices and numeric values to the desired form (either
    categorical indices or normalized numeric values).

6. Tokenizing the data in time to create a pre-tensorized dataset with clear delineations between subjects,
    subject sequence elements, and measurements per sequence element (note that various of these delineations
    may be fully flat/trivial for unnested formats).

7. Tensorizing the data to permit efficient retrieval from disk of subject data for deep-learning modeling
    via PyTorch.

Much like how the entire MEDS ETL pipeline is controlled by a single configuration file, the pre-processing
pipeline is also controlled by a single configuration file, stored in `configs/preprocessing.yaml`. Scripts
leverage this file once again through the [Hydra](https://hydra.cc/) configuration management system. Similar
to the ETL, this pipeline is designed to enable seamless parallelism and efficient use of resources simply by
running multiple copies of the same script on independent workers to process the data in parallel. "Reduction"
steps again need to happen in a single-threaded manner, but these steps are generally very fast and should not
be a bottleneck.

### Tokenization

Tokenization is the process of producing dataframes that are arranged into the sequences that will eventually
be processed by deep-learning methods. Generally, these dataframes will be arranged such that each row
corresponds to a unique subject, with nested list-type columns corresponding either to _events_ (unique
timepoints), themselves with nested, list-type measurements, or to _measurements_ (unique measurements within
a timepoint) directly. Importantly, _tokenized files are generally not ideally suited to direct ingestion by
PyTorch datasets_. Instead, they should undergo a _tensorization_ process to be converted into a format that
permits fast, efficient, scalable retrieval for deep-learning training.

### Tensorization

Tensorization is the process of producing files of the tokenized, normalized sequences that permit efficient,
scalable deep-learning. Here, by _efficiency_, we mean that the file structure and arrangement should permit
the deep learning process to (1) begin smoothly after startup, without a long, data-ingestion phase, (2) be
organized such that individual items (e.g., in a `__getitem__` call) can be retrieved quickly in a manner that
does not inhibit rapid training, and (3) be organized such that CPU and GPU resources are used efficiently
during training. Similarly, by _scalability_, we mean that the three desiderata above should hold true even as
the dataset size grows much larger---while total training time can increase, time to begin training, to
process the data per-item, and CPU/GPU resources required should remain constant, or only grow negligibly,
such as the cost of maintaining a larger index of subject IDs to file offsets or paths (though disk space will
of course increase).

Depending on one's performance needs and dataset sizes, there are 3 modes of deep learning training that can
be used that warrant different styles of tensorization:

#### In-memory Training

This mode of training does not scale to large datasets, and given the parallelizability of the data-loading
phase, may or may not actually be significantly faster than other modes. It is not currently supported in this
repository. **TODO** describe in more detail.

#### Direct Retrieval

This mode of training has the data needed for any given PyTorch Dataset `__getitem__` call retrieved from disk
on an as-needed basis. This mode is extremely scalable, because the entire dataset never need be
loaded or stored in memory in its entirety. When done properly, retrieving data from disk can be done in a
manner that is independent of the total dataset size as well, thereby rendering the load time similarly
unconstrained by total dataset size. This mode is also extremely flexible, because different cohorts can be
loaded from the same base dataset simply by changing which subjects and what offsets within subject data are
read on any given cohort, all without changing the base files or underlying code. However, this mode does
require ragged dataset collation which can be more resource intensive than pre-batched iteration, so it is
slower than the "Fixed-batch retrieval" approach. This mode is what is currently supported by this repository.

#### Fixed-batch Retrieval

In this mode of training, batches are selected once (potentially over many epochs), the items making up those
batches are selected, then their contents are frozen and written to disk in a fully tensorized, padded format.
This enables one to merely load batched data from disk directly onto the GPU during training, which is the
fastest possible way to train a model. However, this mode is less flexible than the other modes, as the
batches are frozen during training and cannot be changed without re-tensorizing the dataset, meaning that
every new cohort for training requires a new tensorization step. This mode is not currently supported by this
repository.

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

## Notes:

You can overwrite the `stages` parameter on the command line to run a dynamic pipeline with just a subset of
options (the `--cfg job --resolve` is just to make hydra show the induced, resolved config instead of trying
to run anything):

```bash
MEDS_transforms on  reusable_interface [$⇡] is 󰏗 v0.0.1 via  v3.12.4 via  MEDS_fns
❯ MEDS_transform-normalization input_dir=foo cohort_dir=bar 'stages=["normalization", "tensorization"]' --cfg job --resolve
```
