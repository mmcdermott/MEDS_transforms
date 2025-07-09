# A Simple Example Package

## Stage: `drop_regex`

This package will define and register a simple new stage for MEDS transform. This stage, defined in the
[`drop_regex/`](src/simple_example_pkg/drop-regex) directory in the source code, will drop all measurements
where the code matches a provided regex. This stage is a map stage -- meaning it applies over each of the data
shards and outputs a new data shard containing the result of the transformation over the input shard. It has
an example of its operation in the [`drop_regex/examples/`](src/simple_example_pkg/drop-regex/examples)
directory, following the stage testing / examples pattern that MEDS-Transforms supports.

## Pipeline: `example_pipeline`

This package also provides a pipeline configuration file, which is located in the
[`pipelines/example_pipeline.yaml`](src/simple_example_pkg/pipelines/example_pipeline.yaml) file. This file
defines a two-stage pipeline, which first drops all codes that start with a capital "H" (via the `drop_regex`
stage, then counts the occurrences of all remaining codes (via the built-in `aggregate_code_metadata` stage).

## Installation and Usage

This package, while an example, can be installed locally via `pip`:

```bash
pip install --quiet -e .
```

After installation, its stage can be referenced by name in MEDS-Transforms pipeline (`drop_regex`) and the
pipeline configuration file can be referenced via the package-specification syntax:
`pkg://simple_example_pkg/pipelines/example_pipeline`.

## Testing

The code in this pipeline can also be tested via `pytest` and `doctest`. It uses both doctests in the source
code and the automated stage testing framework provided by MEDS-Transforms.

To test this package, after installation, you can run `pytest` in the root of the repository, and it should
automatically test the package and its stages. When you do this, it should only run one doctest and one stage
registration test, as it should be for the local package only. This stage has one example test defined, which
uses the regex `^T.*` to drop all codes that start with a capital "T". You can see by comparing the output
data for the stage example (in
[`drop_regex/examples/out_data.yaml`](src/simple_example_pkg/drop-regex/examples/out_data.yaml)) to the
default input data in the [MEDS Testing Helpers
package](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers/blob/main/src/meds_testing_helpers/static_sample_data/simple_static_sharded_by_split.yaml)
that indeed all codes that start with a capital "T" are dropped from the output data.
