# A Simple Example Package

This package will define and register a simple new stage for MEDS transform. This stage, defined in the
[`drop_regex/`](src/simple_example_pkg/drop-regex) directory in the source code, will drop all measurements
where the code matches a provided regex. This stage is a map stage -- meaning it applies over each of the data
shards and outputs a new data shard containing the result of the transformation over the input shard. It has
an example of its operation in the [`drop_regex/examples/`](src/simple_example_pkg/drop-regex/examples)
directory, following the stage testing / examples pattern that MEDS-Transforms supports.
