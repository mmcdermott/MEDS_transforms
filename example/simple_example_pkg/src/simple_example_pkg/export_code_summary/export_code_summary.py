"""Exports a JSON summary of code frequencies from a MEDS dataset.

This stage demonstrates using a custom ``example_class`` on ``Stage.register``. The stage writes
a JSON file (not MEDS-format parquet), so the default ``StageExample.check_outputs`` -- which
expects ``data/*.parquet`` or ``metadata/codes.parquet`` -- cannot validate it. A custom subclass
overrides ``check_outputs`` to compare JSON files instead.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from omegaconf import DictConfig
from yaml_to_disk import yaml_disk

from MEDS_transforms.stages import Stage
from MEDS_transforms.stages.examples import StageExample


@dataclass
class JsonOutputStageExample(StageExample):
    """A StageExample subclass that validates JSON file output instead of MEDS parquet.

    This is useful for stages that produce non-MEDS output formats. The expected output is
    specified in ``out_data.yaml`` as a ``yaml_to_disk`` specification containing the expected
    files; ``check_outputs`` materializes it with ``yaml_disk`` and compares against actual output.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     yaml_fp = Path(tmpdir) / "out_data.yaml"
        ...     _ = yaml_fp.write_text("code_summary.json:\\n  A: 3\\n  B: 1\\n")
        ...     actual_dir = Path(tmpdir) / "actual" / "data"
        ...     actual_dir.mkdir(parents=True)
        ...     _ = (actual_dir / "code_summary.json").write_text('{"A": 3, "B": 1}')
        ...     example = JsonOutputStageExample(
        ...         stage_name="test", scenario_name="s", want_data=yaml_fp,
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
        ...         stage_name="test", scenario_name="s", want_data=yaml_fp,
        ...     )
        ...     example.check_outputs(actual_dir.parent)
        Traceback (most recent call last):
            ...
        AssertionError: JSON mismatch in code_summary.json...
    """

    want_data: Path | None = None

    @classmethod
    def from_dir(cls, stage_name, scenario_name, example_dir, **schema_updates):
        """Parse the example directory, treating ``out_data.yaml`` as a ``yaml_to_disk`` spec."""
        return cls(
            stage_name=stage_name,
            scenario_name=scenario_name,
            want_data=example_dir / "out_data.yaml",
        )

    def check_outputs(self, output_dir, is_resolved_dir=False):
        """Compare expected files (materialized via ``yaml_disk``) against the actual output dir.

        JSON files are compared as parsed objects (order-independent); other files are compared as text.
        """
        if self.want_data is None:
            return

        data_dir = output_dir if is_resolved_dir else output_dir / "data"

        with tempfile.TemporaryDirectory() as expected_root:
            expected_root = Path(expected_root)
            yaml_disk(self.want_data, root_dir=expected_root)

            for expected_fp in sorted(expected_root.rglob("*")):
                if not expected_fp.is_file():
                    continue
                rel = expected_fp.relative_to(expected_root)
                actual_fp = data_dir / rel
                assert actual_fp.is_file(), f"Expected output file {rel} not found in {data_dir}"

                if expected_fp.suffix == ".json":
                    expected_obj = json.loads(expected_fp.read_text())
                    actual_obj = json.loads(actual_fp.read_text())
                    assert expected_obj == actual_obj, (
                        f"JSON mismatch in {rel}:\n  Expected: {expected_obj}\n  Got: {actual_obj}"
                    )
                else:
                    expected_text = expected_fp.read_text().strip()
                    actual_text = actual_fp.read_text().strip()
                    assert expected_text == actual_text, (
                        f"Content mismatch in {rel}:\n  Expected: {expected_text}\n  Got: {actual_text}"
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
