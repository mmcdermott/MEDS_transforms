Removes subjects whose medical timelines are too short to be useful for downstream modeling.

A "measurement" is any row in the dataset (with or without a timestamp), while an "event" is a unique
timestamp for a subject. For example, a subject with 3 lab results at the same time has 3 measurements
but only 1 event. Use `min_events_per_subject` when you care about temporal diversity and
`min_measurements_per_subject` when you care about data volume.
