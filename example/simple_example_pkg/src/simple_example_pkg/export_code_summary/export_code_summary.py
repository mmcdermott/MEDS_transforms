"""Exports a JSON summary of code frequencies from a MEDS dataset.

This stage demonstrates using a custom ``example_class`` on ``Stage.register``. The stage writes
a JSON file (not MEDS-format parquet), so the default ``StageExample.check_outputs`` -- which
expects ``data/*.parquet`` or ``metadata/codes.parquet`` -- cannot validate it. A custom subclass
overrides ``check_outputs`` to compare JSON files instead.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from omegaconf import DictConfig

from MEDS_transforms.stages import Stage
from MEDS_transforms.stages.examples import StageExample


@dataclass
class JsonOutputStageExample(StageExample):
    """A StageExample subclass that validates JSON file output instead of MEDS parquet.

    This is useful for stages that produce non-MEDS output formats. The expected output is
    specified in ``out_data.yaml`` as a yaml_to_disk structure containing JSON files, and
    validation compares those JSON files against the actual stage output.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     yaml_fp = Path(tmpdir) / "out_data.yaml"
        ...     _ = yaml_fp.write_text("code_summary.json:\\n  A: 3\\n  B: 1\\n")
        ...     actual_dir = Path(tmpdir) / "actual" / "data"
        ...     actual_dir.mkdir(parents=True)
        ...     _ = (actual_dir / "code_summary.json").write_text('{"A": 3, "B": 1}')
        ...     example = JsonOutputStageExample(
        ...         stage_name="test", scenario_name="s",
        ...         want_data=yaml_fp, want_metadata=None,
        ...     )
        ...     example.check_outputs(actual_dir.parent)

    Mismatched content raises an error:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     yaml_fp = Path(tmpdir) / "out_data.yaml"
        ...     _ = yaml_fp.write_text("code_summary.json:\\n  A: 3\\n")
        ...     actual_dir = Path(tmpdir) / "actual" / "data"
        ...     actual_dir.mkdir(parents=True)
        ...     _ = (actual_dir / "code_summary.json").write_text('{"A": 5}')
        ...     example = JsonOutputStageExample(
        ...         stage_name="test", scenario_name="s",
        ...         want_data=yaml_fp, want_metadata=None,
        ...     )
        ...     example.check_outputs(actual_dir.parent)
        Traceback (most recent call last):
            ...
        AssertionError: JSON mismatch in code_summary.json...
    """

    # Override to accept Path (directory of expected JSON files) instead of MEDSDataset
    want_data: Path | None = None

    def __post_init__(self):
        if self.want_data is None and self.want_metadata is None:
            raise ValueError("Either want_data or want_metadata must be provided.")
        if self.df_check_kwargs is None:
            self.df_check_kwargs = {"rel_tol": 1e-3, "abs_tol": 1e-5}

    @classmethod
    def from_dir(cls, stage_name, scenario_name, example_dir, **schema_updates):
        """Parse example directory, treating out_data.yaml as a yaml_to_disk path for JSON output."""
        want_data_fp = example_dir / "out_data.yaml"
        in_fp = example_dir / "in.yaml"

        want_data = want_data_fp if want_data_fp.is_file() else None
        in_data = None
        if in_fp.is_file():
            from MEDS_transforms.stages.examples import MEDSDataset

            try:
                in_data = MEDSDataset.from_yaml(in_fp)
            except ValueError:
                in_data = in_fp

        return cls(
            stage_name=stage_name,
            scenario_name=scenario_name,
            want_data=want_data,
            in_data=in_data,
        )

    def check_outputs(self, output_dir, is_resolved_dir=False):
        """Compare expected JSON output against actual output directory.

        Loads expected output from the yaml_to_disk specification and compares against actual files. JSON
        files are compared as parsed objects (order-independent); other files are compared as strings.
        """
        if self.want_data is None:
            return

        data_dir = output_dir if is_resolved_dir else output_dir / "data"

        import yaml

        with open(self.want_data) as f:
            expected_files = yaml.safe_load(f)

        for rel_path, expected_content in expected_files.items():
            actual_fp = data_dir / rel_path
            assert actual_fp.is_file(), f"Expected output file {rel_path} not found in {output_dir}"

            actual_text = actual_fp.read_text().strip()

            if rel_path.endswith(".json"):
                expected_obj = (
                    json.loads(expected_content) if isinstance(expected_content, str) else expected_content
                )
                actual_obj = json.loads(actual_text)
                assert expected_obj == actual_obj, (
                    f"JSON mismatch in {rel_path}:\n  Expected: {expected_obj}\n  Got: {actual_obj}"
                )
            else:
                expected_text = str(expected_content).strip()
                assert expected_text == actual_text, (
                    f"Content mismatch in {rel_path}:\n  Expected: {expected_text}\n  Got: {actual_text}"
                )


@Stage.register(is_metadata=False, example_class=JsonOutputStageExample)
def main(cfg: DictConfig):
    """Reads a MEDS dataset and writes a JSON summary of code frequencies.

    The output is a single ``code_summary.json`` file in the output directory containing a mapping
    from code name to occurrence count across all shards.
    """

    input_dir = Path(cfg.input_dir) / "data"
    output_dir = Path(cfg.stage_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    code_counts: dict[str, int] = {}
    for fp in sorted(input_dir.rglob("*.parquet")):
        df = pl.read_parquet(fp)
        if "code" not in df.columns:
            continue
        counts = df.group_by("code").len().sort("code")
        for row in counts.iter_rows():
            code, count = row
            code_counts[code] = code_counts.get(code, 0) + count

    summary_fp = output_dir / "code_summary.json"
    summary_fp.write_text(json.dumps(code_counts, indent=2, sort_keys=True))
