# add_sequence_number example

This example demonstrates the `example_class` parameter on `Stage.register`.

The stage adds a `seq_num` column to each shard, but `out_data.yaml` intentionally omits it. Under the default
`StageExample`, this would cause a column mismatch error. Because the stage registers with
`example_class=ExtraColumnsStageExample`, the test only validates the columns present in `out_data.yaml` and
tolerates extra columns (`seq_num`) in the actual output.
