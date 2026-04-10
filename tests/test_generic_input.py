"""Tests for non-MEDS (generic) input support in StageExample via yaml_to_disk fallback."""

import tempfile
from pathlib import Path

import yaml

from MEDS_transforms.stages.examples import SIMPLE_STATIC_SHARDED_BY_SPLIT, MEDSDataset, StageExample

_MINIMAL_MEDS_YAML = """\
data/train/0: |-2
  subject_id,time,code,numeric_value
  1,,GENDER//M,
metadata/codes.parquet: |-2
  code,description,parent_codes
  GENDER//M,,
"""


def _make_generic_yaml(tmp: Path) -> Path:
    """Create a non-MEDS in.yaml with CSV and JSON content."""
    in_fp = tmp / "in.yaml"
    in_fp.write_text(
        yaml.dump(
            {
                "raw/patients.csv": "MRN,dob,eye_color\n1195293,06/20/1978,BLUE\n239684,12/28/1980,BROWN",
                "raw/labs.csv": "MRN,lab_date,lab_code,value\n1195293,01/01/2020,HR,80",
                "metadata/.shards.json": {"train/0": [239684], "train/1": [1195293]},
                "event_cfgs.yaml": {
                    "raw/patients": {"eye_color": {"code": '"EYE_COLOR"', "time": None}},
                },
            }
        )
    )
    return in_fp


def _write_out_data_yaml(tmp: Path) -> None:
    """Write a minimal valid MEDS out_data.yaml so StageExample is valid."""
    (tmp / "out_data.yaml").write_text(_MINIMAL_MEDS_YAML)


def test_from_dir_falls_back_to_path_for_non_meds_input():
    """When in.yaml contains non-MEDS keys, from_dir stores the Path instead of a MEDSDataset."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fp = _make_generic_yaml(tmp)
        _write_out_data_yaml(tmp)

        example = StageExample.from_dir("test_stage", "generic", tmp)

        assert isinstance(example.in_data, Path), f"Expected Path, got {type(example.in_data)}"
        assert example.in_data == in_fp


def test_write_for_test_with_generic_input():
    """write_for_test should use yaml_to_disk to write non-MEDS files when in_data is a Path."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_generic_yaml(tmp)
        _write_out_data_yaml(tmp)

        example = StageExample.from_dir("test_stage", "generic", tmp)

        # Write to a separate output directory
        output = tmp / "output"
        output.mkdir()
        example.write_for_test(output)

        # Verify the expected files were written
        assert (output / "raw" / "patients.csv").is_file()
        assert (output / "raw" / "labs.csv").is_file()
        assert (output / "metadata" / ".shards.json").is_file()
        assert (output / "event_cfgs.yaml").is_file()

        # Verify CSV content
        patients = (output / "raw" / "patients.csv").read_text()
        assert "MRN,dob,eye_color" in patients
        assert "1195293" in patients


def test_write_for_test_meds_input_unchanged():
    """Existing MEDS input behavior should be completely unchanged."""
    ds = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        example = StageExample(stage_name="test", want_data=ds, in_data=ds)
        example.write_for_test(tmp)

        assert (tmp / "data" / "train" / "0.parquet").is_file()
        assert (tmp / "data" / "train" / "1.parquet").is_file()
        assert (tmp / "metadata" / "codes.parquet").is_file()
        assert (tmp / "metadata" / "dataset.json").is_file()


def test_str_with_generic_input():
    """__str__ should handle Path in_data without error."""
    want_data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fp = _make_generic_yaml(tmp)

        example = StageExample(stage_name="test", scenario_name="generic", want_data=want_data, in_data=in_fp)
        result = str(example)
        assert "in_data:" in result
        assert "in.yaml" in result


def test_pipeline_cfg_from_dir():
    """from_dir should load pipeline_cfg.yaml when present."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _write_out_data_yaml(tmp)

        pipeline_cfg_fp = tmp / "pipeline_cfg.yaml"
        pipeline_cfg_fp.write_text(yaml.dump({"event_conversion_config_fp": "/path/to/config.yaml"}))

        example = StageExample.from_dir("test_stage", "with_pipeline_cfg", tmp)

        assert example.pipeline_cfg == {"event_conversion_config_fp": "/path/to/config.yaml"}
        assert example.do_use_config_yaml is True


def test_pipeline_cfg_merged_into_cmd_pipeline_cfg():
    """pipeline_cfg keys should appear in cmd_pipeline_cfg alongside stage config."""
    want_data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)

    example = StageExample(
        stage_name="test_stage",
        want_data=want_data,
        stage_cfg={"min_events": 5},
        pipeline_cfg={"event_conversion_config_fp": "/path/to/config.yaml"},
    )

    cfg = example.cmd_pipeline_cfg
    assert cfg is not None
    assert cfg.event_conversion_config_fp == "/path/to/config.yaml"
    assert cfg.stages == [{"test_stage": {"min_events": 5}}]


def test_pipeline_cfg_forces_config_yaml_mode():
    """Setting pipeline_cfg should automatically enable do_use_config_yaml."""
    want_data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)

    example = StageExample(
        stage_name="test",
        want_data=want_data,
        pipeline_cfg={"some_key": "some_value"},
    )
    assert example.do_use_config_yaml is True


def test_pipeline_cfg_absent_no_change():
    """Without pipeline_cfg, behavior is unchanged."""
    want_data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)

    example = StageExample(stage_name="test", want_data=want_data)
    assert example.pipeline_cfg == {}
    assert example.do_use_config_yaml is False
    assert example.cmd_pipeline_cfg is None


def test_pipeline_cfg_in_str():
    """__str__ should display pipeline_cfg when present."""
    want_data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)

    example = StageExample(
        stage_name="test",
        want_data=want_data,
        pipeline_cfg={"event_conversion_config_fp": "/path/to/config.yaml"},
    )
    result = str(example)
    assert "pipeline_cfg:" in result
    assert "event_conversion_config_fp" in result


def test_docgen_with_generic_input():
    """Docgen should render non-MEDS input as a YAML code block."""
    from MEDS_transforms.stages.docgen import _format_example

    want_data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fp = _make_generic_yaml(tmp)

        example = StageExample(
            stage_name="test_stage",
            scenario_name="generic",
            want_data=want_data,
            in_data=in_fp,
        )
        output = _format_example("test_stage", example)
        assert "**Input files:**" in output
        assert "```yaml" in output
        assert "raw/patients.csv" in output
        # Should NOT contain _format_dataset output
        assert "**Input data:**" not in output
