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
2. A flexible ETL for extracting MEDS data from a variety of source formats.
3. A pre-processing pipeline that can be used for models that require:
   \- Filtering data to only include patients with a certain number of events
   \- Filtering events to only include those whose codes occur a certain number of times
   \- Removing numerical values that are more than a certain number of standard deviations from the mean on a
   per-code basis.
   \- Normalizing numerical values on a per-code basis.
   \- Modeling data in a 3D format, with patients X unique timestamps X codes & numerical values per
   timestamp.

## Installation

For now, clone this repository and run `pip install -e .` from the repository root.

### ETL

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
   \- First, each input shard is converted to the MEDS flat format and split into sub patient-level shards.
   \- Second, the appropriate sub patient-level shards are joined and re-organized into the final
   patient-level shards. This method ensures that we minimize the amount of read contention on the input
   shards during the join process and can maximize parallel throughput, as (theoretically, with sufficient
   workers) all input shards can be sub-sharded in parallel and then all output shards can be joined in
   parallel.

The ETL scripts all use [Hydra](https://hydra.cc/) for configuration management, leveraging the shared
`configs/extraction.yaml` file for configuration. The user can override any of these settings in the normal
way for Hydra configurations.

#### Input Event Extraction

Input events extraction configurations are defined through a simple configuration file language, stored in
YAML form on disk, which specified for a collection of events how the individual rows from the various input
dataframes should be parsed into different event formats. The YAML file stores a simple dictionary with the
following structure:

```yaml
$INPUT_FILE_STEM:
    $EVENT_NAME:
        code: $CODE
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

### Pre-processing
