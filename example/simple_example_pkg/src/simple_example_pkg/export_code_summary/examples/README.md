# export_code_summary example

This example demonstrates the `example_class` parameter on `Stage.register`.

The stage writes a JSON file (`code_summary.json`), not MEDS-format parquet. The default
`StageExample.check_outputs` expects `data/*.parquet` or `metadata/codes.parquet` and would fail
here. By registering with `example_class=JsonOutputStageExample`, the stage uses a subclass that
validates JSON output instead.

The `out_data.yaml` file here is a `yaml_to_disk` specification describing the expected files.
`JsonOutputStageExample.check_outputs` materializes it into a temporary directory using
`yaml_disk`, then compares each expected file against the actual stage output (JSON files are
compared as parsed objects; other files as text).
