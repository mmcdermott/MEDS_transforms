# export_code_summary example

This example demonstrates the `example_class` parameter on `Stage.register`.

The stage writes a JSON file (`code_summary.json`), not MEDS-format parquet. The default
`StageExample.check_outputs` expects `data/*.parquet` or `metadata/codes.parquet` and would fail
here. By registering with `example_class=JsonOutputStageExample`, the stage uses a subclass that
validates output by comparing JSON files via `yaml_to_disk` instead.

The `out_data.yaml` file here is a `yaml_to_disk` specification that describes the expected JSON
output, not a MEDS dataset.
