# `deduplicate_events` example

Demonstrates the `on_conflict: first` policy on a single shard with three kinds of duplicate-shaped rows:

- Two identical `(239684, ?, EYE_COLOR//BROWN, ?)` rows — true duplicate, collapses to one.
- Two identical `(239684, "05/11/2010, 17:41:51", HR, 102.6)` rows — true duplicate, collapses to one.
- Two `(239684, "05/11/2010, 17:41:51", HR, ?)` rows with different `numeric_value` (102.6 vs 103.5) — not a duplicate under the default `by` (which includes `numeric_value`); both rows survive.

Under `on_conflict: error` the first two collapses still happen; only genuine `by`-key disagreements raise. To see that, swap in the `numeric_value` from the third row above so two rows share `(subject_id, time, code, numeric_value)` but differ on, say, `text_value`, and switch back to `on_conflict: error`.
