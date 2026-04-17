"""Regression test for https://github.com/mmcdermott/MEDS_transforms/issues/321.

Confirms ``reshard_to_split`` runs correctly when invoked through
``MEDS_transform-stage`` as the only stage in a pipeline (single-stage mode).
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from meds import DataSchema, subject_splits_filepath


def _write_dataset(input_dir: Path) -> None:
    (input_dir / "data" / "train").mkdir(parents=True, exist_ok=True)
    (input_dir / "data" / "tuning").mkdir(parents=True, exist_ok=True)
    (input_dir / "data" / "held_out").mkdir(parents=True, exist_ok=True)
    (input_dir / "metadata").mkdir(parents=True, exist_ok=True)

    base_time = datetime(2020, 1, 1, tzinfo=UTC)

    def _write(path: Path, subjects: list[int]) -> None:
        # Use a real timestamp column so reshard_to_split's sort by DataSchema.time_name
        # hits a well-typed column, matching production MEDS data.
        rows = [
            {
                "subject_id": s,
                "time": base_time.replace(day=1 + i),
                "code": "A",
                "numeric_value": None,
            }
            for s in subjects
            for i in range(2)
        ]
        (pl.DataFrame(rows, schema_overrides={DataSchema.time_name: pl.Datetime("us")}).write_parquet(path))

    _write(input_dir / "data" / "train" / "0.parquet", [1, 2, 3, 4])
    _write(input_dir / "data" / "tuning" / "0.parquet", [5, 6])
    _write(input_dir / "data" / "held_out" / "0.parquet", [7])

    splits = pl.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6, 7],
            "split": ["train"] * 4 + ["tuning"] * 2 + ["held_out"],
        }
    )
    splits.write_parquet(input_dir / subject_splits_filepath)


def test_reshard_to_split_runs_as_only_stage() -> None:
    """``reshard_to_split`` as the sole stage in a pipeline produces correct sub-shards."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_dir = tmp / "input"
        output_dir = tmp / "output"
        _write_dataset(input_dir)

        (tmp / "pipeline.yaml").write_text(
            f"input_dir: {input_dir}\n"
            f"output_dir: {output_dir}\n"
            "stages:\n"
            "  - reshard_to_split:\n"
            "      n_subjects_per_shard: 2\n"
        )

        result = subprocess.run(
            [
                "MEDS_transform-stage",
                str(tmp / "pipeline.yaml"),
                "reshard_to_split",
                "stage=reshard_to_split",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"Single-stage reshard failed:\n{result.stderr}\n{result.stdout}"

        # When reshard_to_split is the only (final) stage, output lands in ``output_dir/data``.
        shards_json = output_dir / "data" / ".shards.json"
        assert shards_json.is_file()

        import json

        assignment = json.loads(shards_json.read_text())
        assert set(assignment.keys()) == {"train/0", "train/1", "tuning/0", "held_out/0"}
        # 4 train subjects split 2-2, 2 tuning in one shard, 1 held_out in one shard.
        # Compare as sets: the stage computes shard membership from parquet reads, so within-shard
        # ordering is an implementation detail that shouldn't gate the regression test.
        assert set(assignment["train/0"] + assignment["train/1"]) == {1, 2, 3, 4}
        assert len(assignment["train/0"]) == 2 and len(assignment["train/1"]) == 2
        assert set(assignment["tuning/0"]) == {5, 6}
        assert set(assignment["held_out/0"]) == {7}

        # Every produced shard should be a valid parquet readable by polars.
        for shard_name in assignment:
            shard_fp = output_dir / "data" / f"{shard_name}.parquet"
            df = pl.read_parquet(shard_fp)
            assert df.height > 0, f"empty shard: {shard_fp}"
            assert set(df["subject_id"].unique().to_list()) == set(assignment[shard_name])
