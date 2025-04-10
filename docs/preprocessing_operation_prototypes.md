# MEDS Pre-processing Operation Prototypes STILL IN PROGRESS

[!NOTE]
This document is currently aspirational, not yet implemented. Some functions in these patterns are
implemented, but not universally.

To support communal development and sharing of pre-processing operations, MEDS defines a set of core operation
"prototypes", which are extensible, reusable operations that can be applied to MEDS datasets in a variety of
circumstances to accomplish diverse, yet common, pre-processing tasks. The intent with these prototypes is
both that users can leverage these pre-built operations to quickly and easily accomplish common tasks, and
that they can _extend_ the set of supported operations within the broader framework using these prototypes to
support new operations to accelerate their own development and streamline the adoption of their innovations by
the broader community.

Note that, pursuant to the [core MEDS terminology](terminology.md), we will use "code" to refer to the unique
sets (with `null`s allowed) of `code` and all `code_modifier` columns. All operations should be presumed to be
potentially parametrized by the datasets list of code modifier columns.

A note on terminology: We will use the term "removing data" to refer to operations that fully drop data from
the record, retaining no notion of the corresponding data occurring in the dataset. Operations that remove
data will result in smaller overall datasets (either in number of patients or number of measurements). We will
use the term "occluding data" to refer to operations that set data to `UNK`, `None`, or `np.NaN`, but retain
that there was _some_ data in the dataset originally. Operations that occlude data will result in the same
size dataset in terms of number of patients, number of measurements, etc., but will not have the same degree
of data granularity or information content. Occlud operations will typically *not* be reversible, but will
include a boolean indicator identifying that data was definitively occluded.

## Filtering Prototypes (a.k.a. Match and Revise)

A subset of the prototypes listed below can be modified to only be applied to a subset of the data. These
subsets can be based on patient level criteria (e.g., patients who meet certain criteria) or via code filters
(e.g., to only apply a certain value extraction regex to codes that match a certain pattern), with the
results being merged into the output dataset in a consistent manner. Currently, these capabilities are only
planned, not yet implemented.

## Prototypes

### Transform Codes (just codes, not patient data!)

This operation is used to perform a static re-mapping over the allowed codes in a MEDS dataset, typically in
preparation for mapping that transformation out across the patient data by code.

##### Operation Steps

1. Add new information or transform existing columns in an existing `metadata/codes.parquet` file. Note that
    `code` or `code_modifier` columns should _not_ be modified in this step as that will break the linkage
    with the patient data.

##### Parameters

1. What function should be applied to each code row.

##### Status

Individual functions are supported, but the operation as a prototypical paradigm is not yet implemented.

##### Currently Supported Operations

Functions:

1. Assign vocabulary indices to codes. See `src/MEDS_transforms/get_vocabulary.py`

### Collect metadata about code realizations in patient data

This operation is used to produce summary information about the realizations of any unique code in the data,
such as the number of times a code occurs, the mean or variance of numerical values that are associated with a
code, etc. This operation can be applied over all data, or across patient groups or cohorts (saved into
separate files per patient group -- each output file is only grouped by code, not by patient group for
simplicity).

##### Operation Steps

1. Per-shard, filter the pateint data to satisfy desired set of patient or other data criteria.
2. Per-shard, group by code and collect some aggregate statistics. Optionally also compute statistics across
    all codes.
3. Reduce the per-shard aggregate files into a unified `metadata/codes.parquet` file.
4. Optionally merge with static per-code metadata from prior steps.

##### Parameters

1. What (if any) patient data filters should be applied prior to aggregation.
2. What aggregation functions should be applied to each code. Each aggregation function must specify both a
    _mapper_ function that computes aggregate data on a per-shard basis and a _reducer_ function that
    combines different shards together into a single, unified metadata file.
3. Whether or not aggregation functions should be computed over all raw data (the "null" code case).

##### Status

This operation is partially implemented as a prototype, but is not yet fully functional, as it lacks support
for patient-data filters prior to aggregation.

##### Currently supported operations

Patient Filters: **None**

Functions:

1. Various aggregation functions; see `src/MEDS_transforms/aggregate_code_metadata.py` for a list of supported
    functions.

##### Planned Future Operations

None at this time. To request a new operation, please open a GitHub issue.

### Filtering the Dataset

There are a few modes of filtering data from MEDS datasets that are configured as separate prototypes. These
include:

1. Filtering patients wholesale based on aggregate, patient-level criteria (e.g., number of events, etc.)
2. Filtering the data to only include patient data that matches some cohort specification (meaning removing
    data that is not within pre-identified ranges of time on a per-patient basis).
3. Filtering individual measurements from the data based on some criteria (e.g., removing measurements that
    have codes that are not included in the overall vocabulary, etc.).

#### Filtering Patients

##### Operation Steps

1. Per-shard, aggregate data per-patient and compute some aggregate criteria.
2. Remove all data corresponding to patients on the basis of the resulting criteria.
3. Return the filtered dataset, in the same format as the original, but with only the remaining patients.

##### Parameters

1. What aggregation functions should be applied to each patient.
2. What criteria should be used based on those aggregations to filter patients.

These parameters may be specified with a single variable (e.g., `min_events_per_patient` indicates we need to
compute the number of unique timepoints per patient and impose a minimum threshold on that number).

##### Status

This operation is only implemented through two concrete functions, not a generalizable prototype in
`src/MEDS_transforms/filter_patients_by_length.py`.

##### Currently supported operations

1. Filtering patients by the number of events (unique timepoints) in their record.
2. Filtering patients by the number of measurements in their record.

##### Planned Future Operations

None at this time. To request a new operation, please open a GitHub issue.

#### Filtering Measurements

This operation assumes that any requisite aggregate, per-code information is pre-computed and can be joined in
via a `metadata/codes.parquet` file.

##### Operation Steps

1. Per-shard, join the data, if necessary, to the provided, global `metadata/codes.parquet` file.
2. Apply row-based criteria to each measurement to determine if it should be retained or removed.
3. Return the filtered dataset, in the same format as the original, but with only the measurements to be
    retained.

##### Parameters

1. What criteria should be used to filter measurements.
2. What, if any, columns in the `metadata/codes.parquet` file should be joined in to the data.

##### Status

This operation is supported as a partial prototype, through the
`src/MEDS_transforms/filter_measurements.py` file. It needs extension to reach a full prototype status,
but supports such extension relatively natively.

##### Currently supported operations

Currently, measurements can be filtered on the basis of `min_patients_per_code` and `min_occurrences_per_code`
thresholds, which are read from the `metadata/codes.parquet` file via the `code/n_patients` and
`code/n_occurrences` columns, respectively.

##### Planned Future Operations

None at this time. To request a new operation, please open a GitHub issue.

### Transforming Features within Measurements

These prototypes or functional patterns are for transforming features within measurements. Critically, they
leave the output dataset in the same length and in the same order as the input dataset, and only transform
features. For operations that change the length or order (within the mandated `patient_id` and `timepoint`
order), see the "Transforming Measurements within Events" section.

**TODO** Add or merge in the following:

1. Normalizing numerical values (this is currently implemented with `normalization.py`).
2. Extract numerical values from text (e.g., extracting a number from a string).

#### Occluding Features within Measurements

This operation assumes that any requisite aggregate, per-code information is pre-computed and can be joined in
via a `metadata/codes.parquet` file.

**TODO** This is not really a prototype, but is really a single function, or a subset of a prototype. IT has
functionally the same API as numerical value normalization, with the modification that the indicator columns
are added and this function is not reversible.

##### Operation Steps

1. Per-shard, join the data, if necessary, to the provided, global `metadata/codes.parquet` file.
2. Apply row-based criteria to each measurement to determine if individual features should be occluded or
    retained in full granularity.
3. Set occluded data to the occlusion target (typically `"UNK"`, `None`, or `np.NaN`) and add an indicator
    column indicating occlusion status.

##### Parameters

1. What criteria should be used to occlude features.
    - Relatedly, what occlusion value should be used for occluded features.
    - Relatedly, what the name of the occlusion column should be (can be set by default for features).
2. What, if any, columns in the `metadata/codes.parquet` file should be joined in to the data.

##### Status

This operation is only supported through the single `filter_outliers_fntr` function in
`src/MEDS_transforms/filter_measurements.py`. It is not yet a general prototype.

##### Currently supported operations

1. Occluding numerical values if they take a value more distant from the code's mean by a specified number
    of standard deviations.

### Transforming Measurements within Events

These aren't implemented yet, but are planned:

1. Re-order measurements within the event ordering.
2. Split measurements into multiple measurements in a particular order and via a particular functional form.
    E.g.,
    - Performing ontology expansion
    - Splitting a multi-faceted measurement (e.g., blood pressure recorded as `"120/80"`) into multiple
        measurements (e.g., a systolic and diastolic blood pressure measurement with values `120` and `80`).

## Requesting New Prototypes

To request or suggest a new prototypical paradigm, please open a GitHub issue. In that issue, please include a
description of the desired operation in the format used for the operations above, following the below
template:

```markdown
### NAME

Describe the operation in natural language.

##### Operation Steps
Describe the rough API that this operation would take, as a configurable prototype.

##### Parameters
Describe how this operation would be controlled in pipelines by the user. This will ultimately map into
configuration parameters.

##### Status
Describe the current status of this operation. It may, generally speaking, either be fully unsupported, have
realizations of select funcctions supported, but not a general prototype, or be supported either partially or
fully as a prototype.

##### Currently supported operations
Describe what specific realizations of this operation as a prototypes are (e.g., what options the user can
select to realize different functions within this prototype).

##### Planned Future Operations
ADD TEXT HERE
```
