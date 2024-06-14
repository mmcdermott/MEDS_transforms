# MEDS Pre-processing Operation Prototypes STILL IN PROGRESS

To support communal development and sharing of pre-processing operations, MEDS defines a set of core operation
"prototypes", which are extensible, reusable operations that can be applied to MEDS datasets in a variety of
circumstances to accomplish diverse, yet common, pre-processing tasks. The intent with these prototypes is
both that users can leverage these pre-built operations to quickly and easily accomplish common tasks, and
that they can _extend_ the set of supported operations within the broader framework using these prototypes to
support new operations to accelerate their own development and streamline the adoption of their innovations by
the broader community.

## Core Prototypes

### Collect & Aggregate Metadata

#### `collect_code_metadata`

This prototype is for summarizing MEDS data by code (and code modifier columns) and collecting aggregate
information across diverse axes over the entire dataset (or a subset of shards of the data, such as all those
shards in the train set).

TODO: Describe the operation in more detail.

### Filter the dataset

#### `remove_patients`

For removing patients who fail to meet some criteria from the dataset.

#### `remove_measurements`

For removing measurements that fail to meet some criteria from the dataset.

### Uncategorized as of yet.

#### `occlude_outliers`

For occluding (setting to `None` or `np.NaN`) features observed about events within the data. This is
typically used for removing outlier numerical values, but could in theory be used on other features as well,
though that would require additional development.

#### `reorder_measurements`

Some pipelines desire a specific order of measurements within the broader per-patient event order (meaning the
order as implied by unique timestamps).

#### `extract_numeric_values`

These prototypes are for extracting numeric values from other columns in the dataset, most notably `text` or
`categorical` value columns.

#### `extract_categorical_values`

These prototypes are for extracting numeric values from other columns in the dataset, most notably `text` or
`categorical` value columns.

## Possible Future Prototypes

### `remove_events`

For filtering unique timestamps based on some criteria.

### `filter_to_cohort`

For filtering the dataset to only include data matching some cohort specification, as defined by a dataframe
of patient IDs and start/end timestamps. This is not currently occluded as it can often happen trivially
during the dataloading stage for machine learning models, but for true supervised training, it may be useful
so that train-set pre-processing parameters can be fit specific to the cohort-specific train set, rather than
the general train set.
