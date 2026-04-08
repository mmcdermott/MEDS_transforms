These examples show `aggregate_code_metadata` in several contexts:

- **on_raw_static_data** and **with_not_split_defined_shards** -- comprehensive aggregation with
    quantiles, counts, and distribution statistics on the full static sample data.
- **in_example_pipeline/fit_normalization** -- collects only the statistics needed for z-score
    normalization (`n_occurrences`, `n_subjects`, `sum`, `sum_sqd`).
- **in_example_pipeline/fit_outlier_detection** -- collects only the statistics needed for outlier
    detection (`n_occurrences`, `sum`, `sum_sqd`).
