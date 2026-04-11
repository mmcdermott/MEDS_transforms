# A Simple Example Package

## Stage: `drop_regex`

This package will define and register a simple new stage for MEDS transform. This stage, defined in the
[`drop_regex/`](src/simple_example_pkg/drop-regex) directory in the source code, will drop all measurements
where the code matches a provided regex. This stage is a map stage -- meaning it applies over each of the data
shards and outputs a new data shard containing the result of the transformation over the input shard. It has
an example of its operation in the [`drop_regex/examples/`](src/simple_example_pkg/drop-regex/examples)
directory, following the stage testing / examples pattern that MEDS-Transforms supports.

## Stage: `csv_to_meds`

This stage, defined in the [`csv_to_meds/`](src/simple_example_pkg/csv_to_meds) directory, demonstrates how
to build a stage that takes **non-MEDS input** -- raw CSV files and a JSON shard map -- and converts them into
MEDS-formatted parquet output. This is the pattern used by packages like
[MEDS-Extract](https://github.com/mmcdermott/MEDS_extract), whose stages operate on raw source data rather
than pre-existing MEDS datasets.

Because the standard map/reduce stage types expect MEDS-formatted input shards, this stage is registered as a
**MAIN** stage (the function is named `main`), which gives it the full Hydra config and complete control over
its own I/O. It reads CSVs from the input directory, pivots measurement columns into MEDS `(code, numeric_value)` pairs, and writes parquet shards according to the shard map.

The stage's [example](src/simple_example_pkg/csv_to_meds/examples/simple) uses a non-MEDS `in.yaml` that
contains raw CSV files and a JSON shard map instead of MEDS-formatted data. This exercises the `yaml_to_disk`
fallback in the `StageExample` framework -- when `in.yaml` cannot be parsed as a `MEDSDataset`, the framework
stores the raw file path and uses `yaml_to_disk` to write the files to the test input directory. This means
the same example infrastructure used for MEDS stages (automated testing, documentation generation) also works
for non-MEDS input stages.

## Stage: `add_sequence_number`

This stage, defined in the
[`add_sequence_number/`](src/simple_example_pkg/add_sequence_number) directory, demonstrates how to use a
**custom `StageExample` subclass** via the `example_class` parameter on `Stage.register`. This is useful for
downstream packages whose stages produce output columns beyond the standard MEDS schema, or that need custom
output validation logic.

The stage adds a `seq_num` column (a per-subject event counter) to each shard. Its expected output
(`out_data.yaml`) intentionally omits this extra column. Under the default `StageExample`, this would cause a
column mismatch error during testing. Instead, the stage registers with
`example_class=ExtraColumnsStageExample`, a subclass that overrides `check_outputs` to select only the
expected columns before comparison, tolerating any extra columns in the actual output.

This pattern is useful for any downstream package that needs to customize how test examples validate stage
output -- for example, tolerating extra columns, using non-MEDS output formats, or applying domain-specific
comparison logic.

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
