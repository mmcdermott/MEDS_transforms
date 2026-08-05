# `deduplicate_events` example

Demonstrates the `on_conflict: first` policy on a single shard with three kinds of duplicate-shaped rows:

- Two identical `(239684, null, EYE_COLOR//BROWN, null)` rows — true duplicate, collapses to one.
- Two identical `(239684, "05/11/2010, 17:41:51", HR, 102.6)` rows — true duplicate, collapses to one.
- Two `(239684, "05/11/2010, 17:41:51", HR, ...)` rows with different `numeric_value` (102.6 vs 103.5) — not a duplicate under the default `by` (which is the full MEDS `DataSchema` field set, including `numeric_value`); both rows survive.

Under `on_conflict: error` the first two collapses still happen; only genuine `by`-key disagreements raise. To see that, attach a non-schema column (e.g. `annotation: ["from-EHR", "from-claims"]`) to two otherwise-identical rows and switch back to `on_conflict: error` — the default `by` excludes `annotation`, so the two rows are seen as the same event with two conflicting annotations.
