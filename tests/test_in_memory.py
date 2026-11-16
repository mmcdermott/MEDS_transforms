"""Tests of the in-memory store's integration with the package's IO helpers (see `MEDS_transforms.in_memory`).

Each test lays out a small "pipeline output" with one stage's ``data/`` directory held in memory, and checks
that the helpers that list, carry, prune, overwrite and read shards treat it as they would a directory on
disk, while everything outside it stays on disk.
"""

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.in_memory import FrameStore, in_memory_mode
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap
from MEDS_transforms.runner import run_stage_in_memory
from MEDS_transforms.stage_outputs import carry_forward, prune_stale_outputs


def _shard(value: int) -> pl.DataFrame:
    return pl.DataFrame({"subject_id": [value]})


def _write_disk_shards(root: Path, shards: dict[str, int]) -> None:
    for rel, value in shards.items():
        fp = root / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        _shard(value).write_parquet(fp)


def test_get_missing_entry_raises(tmp_path: Path):
    store = FrameStore()
    store.add_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="No in-memory frame"):
        store.get(tmp_path / "missing.parquet")


def test_carry_forward_between_memory_and_disk(tmp_path: Path):
    """Shards carry from disk into memory, between in-memory roots, and from memory back onto disk."""
    disk_in, mem_a, mem_b, disk_out = (tmp_path / n for n in ("in", "a", "b", "out"))
    _write_disk_shards(disk_in, {"train/0.parquet": 0, "train/1.parquet": 1})

    with in_memory_mode() as store:
        store.add_root(mem_a)
        store.add_root(mem_b)

        # Disk -> memory: entries refer to the disk files.
        assert len(carry_forward(disk_in, mem_a, mirror=True)) == 2
        assert store.get(mem_a / "train" / "0.parquet") == disk_in / "train" / "0.parquet"
        # Already current, so a second mirror changes nothing.
        assert carry_forward(disk_in, mem_a, mirror=True) == []

        # Memory -> memory: entries are shared; a stale destination entry is dropped by a mirror.
        store.put(mem_b / "train" / "9.parquet", _shard(9))
        assert len(carry_forward(mem_a, mem_b, mirror=True)) == 2
        assert sorted(p.as_posix() for p in store.files(mem_b)) == ["train/0.parquet", "train/1.parquet"]
        assert carry_forward(mem_a, mem_b, mirror=True) == []

        # Memory -> disk: frames are written out, disk-backed entries are linked or copied.
        store.put(mem_b / "train" / "1.parquet", _shard(11))
        carry_forward(mem_b, disk_out, mirror=True)
        # A mirror onto disk always rewrites what came from memory.
        assert len(carry_forward(mem_b, disk_out, mirror=True)) == 2

    assert pl.read_parquet(disk_out / "train" / "0.parquet")["subject_id"].to_list() == [0]
    assert pl.read_parquet(disk_out / "train" / "1.parquet")["subject_id"].to_list() == [11]
    assert not mem_a.exists() and not mem_b.exists()


def test_prune_stale_outputs_in_memory(tmp_path: Path):
    with in_memory_mode() as store:
        store.add_root(tmp_path)
        for name in ("0", "1"):
            store.put(tmp_path / "train" / f"{name}.parquet", _shard(int(name)))
        removed = prune_stale_outputs(tmp_path, [tmp_path / "train" / "0.parquet"])
        assert [p.name for p in removed] == ["1.parquet"]
        assert sorted(p.as_posix() for p in store.files(tmp_path)) == ["train/0.parquet"]


def test_rwlock_wrap_in_memory_overwrites_without_locks(tmp_path: Path):
    in_fp = tmp_path / "in.parquet"
    _shard(1).write_parquet(in_fp)
    out_fp = tmp_path / "stage" / "data" / "0.parquet"

    def double(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(pl.col("subject_id") * 2)

    with in_memory_mode() as store:
        store.add_root(out_fp.parent)
        assert rwlock_wrap(in_fp, out_fp, read_df, write_df, double)
        # Exists, so it is skipped...
        assert not rwlock_wrap(in_fp, out_fp, read_df, write_df, double)
        # ...unless overwriting.
        assert rwlock_wrap(in_fp, out_fp, read_df, write_df, double, do_overwrite=True)
        assert read_df(out_fp).collect()["subject_id"].to_list() == [2]

    assert not (tmp_path / "stage").exists()


def test_run_stage_in_memory_adopts_shards_written_to_disk(tmp_path: Path):
    """A stage that writes its shards straight to disk (its own write function) still feeds the next."""
    output_dir = tmp_path / "out"
    data_dir = output_dir / "custom" / "data"

    def fake_run(pipeline_config_fp, stage_name, overrides):
        _write_disk_shards(data_dir, {"train/0.parquet": 5})

    with in_memory_mode() as store, patch("MEDS_transforms.runner.run_stage_in_process", fake_run):
        store.add_root(output_dir / "prior" / "data")
        store.put(output_dir / "prior" / "data" / "train" / "0.parquet", _shard(0))

        run_stage_in_memory(store, "pipeline.yaml", "custom", "prior", [], output_dir)

        assert store.read(data_dir / "train" / "0.parquet").collect()["subject_id"].to_list() == [5]
        # The prior stage's in-memory data is no longer needed, so it is dropped.
        assert not store.owns(output_dir / "prior" / "data" / "train" / "0.parquet")
