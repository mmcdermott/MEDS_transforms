Reorganizes a MEDS dataset from its original shard layout into train/tuning/held_out splits.

The input shards may contain subjects from any split. This stage reads `subject_splits` metadata
to assign each subject to the correct split, then writes new shards with at most
`n_subjects_per_shard` subjects each. The output directory structure has one subdirectory per split,
each containing numbered shard files.

This is typically one of the first stages in a pipeline, run right after initial data ingestion.
