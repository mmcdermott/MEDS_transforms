"""Tests that a stage launch leaves a complete, current MEDS dataset in its output directory.

These run the real ``MEDS_transform-stage`` launcher in a subprocess, as the pipeline runner does, to check:

- A launch does not return while another process (e.g. a concurrent launch of the same command) still holds
  one of its shards, and takes the shard over if that process gives it up without writing it.
- Shards left over from an earlier run are removed.
- The metadata a data stage does not compute is carried over from its input.
"""

import subprocess
import tempfile
import time
from pathlib import Path

from filelock import FileLock

from MEDS_transforms.stages.discovery import get_all_registered_stages

PIPELINE_YAML = """
input_dir: {input_dir}
output_dir: {output_dir}

stages:
  - filter_subjects:
      min_events_per_subject: 5
"""


def _setup(root: Path) -> tuple[Path, Path, list[str]]:
    input_dir, output_dir = root / "input", root / "output"
    input_dir.mkdir()
    get_all_registered_stages()["filter_subjects"].load().test_cases["."].write_for_test(input_dir)

    pipeline_fp = root / "pipeline.yaml"
    pipeline_fp.write_text(PIPELINE_YAML.format(input_dir=input_dir, output_dir=output_dir))
    cmd = ["MEDS_transform-stage", str(pipeline_fp), "filter_subjects", "stage=filter_subjects"]
    return input_dir, output_dir, cmd


def _rel_files(root: Path) -> set[str]:
    return {fp.relative_to(root).as_posix() for fp in root.rglob("*") if fp.is_file()}


def test_stage_waits_for_shards_held_by_other_processes():
    with tempfile.TemporaryDirectory() as tmpdir:
        _, output_dir, cmd = _setup(Path(tmpdir))

        held_fp = output_dir / "data" / "train" / "0.parquet"
        held_fp.parent.mkdir(parents=True)
        lock = FileLock(str(held_fp.with_suffix(".parquet.lock")))
        lock.acquire()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait until the launch has written every shard it could, then check it is still waiting.
            deadline = time.monotonic() + 60
            while (
                time.monotonic() < deadline and not (output_dir / "data" / "held_out" / "0.parquet").exists()
            ):
                time.sleep(0.1)
            time.sleep(2)
            assert proc.poll() is None, "The stage returned while another process held one of its shards."
            assert not held_fp.exists()
        finally:
            # Release without writing the shard, as a crashed worker would.
            lock.release()

        stdout, stderr = proc.communicate(timeout=60)
        assert proc.returncode == 0, f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
        assert held_fp.is_file(), "The stage did not take over the shard once its lock was released."


def test_stage_removes_stale_shards_and_carries_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir, output_dir, cmd = _setup(Path(tmpdir))

        stale_fp = output_dir / "data" / "train" / "99.parquet"
        stale_fp.parent.mkdir(parents=True)
        stale_fp.write_bytes((input_dir / "data" / "train" / "0.parquet").read_bytes())
        stale_metadata_fp = output_dir / "metadata" / "old.json"
        stale_metadata_fp.parent.mkdir(parents=True)
        stale_metadata_fp.write_text("{}")

        out = subprocess.run(cmd, capture_output=True, check=False)
        assert out.returncode == 0, f"stdout:\n{out.stdout.decode()}\nstderr:\n{out.stderr.decode()}"

        assert _rel_files(output_dir / "data") == _rel_files(input_dir / "data")
        assert _rel_files(output_dir / "metadata") == _rel_files(input_dir / "metadata")
        for rel in _rel_files(input_dir / "metadata"):
            assert (output_dir / "metadata" / rel).samefile(input_dir / "metadata" / rel)
