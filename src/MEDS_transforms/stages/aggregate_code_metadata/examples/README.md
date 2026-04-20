These examples show `aggregate_code_metadata` in several contexts:

- **on_raw_static_data** and **with_not_split_defined_shards** -- comprehensive aggregation with
    the default quantiles (`0.25 / 0.5 / 0.75`), counts, and distribution statistics on the full
    static sample data.
- **with_custom_quantiles** -- same data but with user-chosen `quantiles: [0.1, 0.9]`. The
    `values/quantiles` struct schema is generated from the stage config, so non-default
    `quantiles` round-trip through example validation (see issue #342).
- **in_example_pipeline/fit_normalization** -- collects only the statistics needed for z-score
    normalization (`n_occurrences`, `n_subjects`, `sum`, `sum_sqd`).
- **in_example_pipeline/fit_outlier_detection** -- collects only the statistics needed for outlier
    detection (`n_occurrences`, `sum`, `sum_sqd`).
