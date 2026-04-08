Computes per-code summary statistics over an entire MEDS dataset using a map-reduce pattern.

The map step scans each data shard and computes local aggregates (counts, sums, etc.) for every code.
The reduce step merges these partial results into a single `codes.parquet` metadata file. The resulting
metadata powers downstream stages like normalization (which needs mean/std), outlier detection (which
needs mean/std), and vocabulary fitting (which needs code counts).

Common aggregations include:

- `code/n_occurrences`, `code/n_subjects` -- how often each code appears and for how many subjects.
- `values/sum`, `values/sum_sqd` -- raw sums for computing mean and standard deviation.
- `values/min`, `values/max`, `values/quantiles` -- distribution summaries.

Set `do_summarize_over_all_codes: True` to also produce a summary row aggregated across all codes.
