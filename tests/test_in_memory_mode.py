"""End-to-end tests for in-memory execution mode (Phase 0).

These tests verify that:

- A single ``rwlock_wrap`` call under ``in_memory_mode`` reads from and writes to the registry
  without touching the filesystem.
- Two MAP stages can be chained in-memory by setting the second stage's ``input`` key to the
  first stage's ``output`` key.
- The in-memory result is value-equal to the on-disk result computed from the same inputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from MEDS_transforms.compute_modes import FrameRegistry, in_memory_mode
from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap


def _double_a(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(pl.col("a") * 2)


def _filter_positive(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(pl.col("a") > 0)


def test_single_stage_in_memory_bypasses_disk():
    """rwlock_wrap under in_memory_mode reads/writes the registry, not the filesystem."""
    registry = FrameRegistry()
    in_key = Path("/virtual/in/shard_0.parquet")
    out_key = Path("/virtual/out/shard_0.parquet")
    registry.put(in_key, pl.LazyFrame({"a": [1, 2, 3]}))

    with in_memory_mode(registry):
        ran = rwlock_wrap(in_key, out_key, read_df, write_df, compute_fn=_double_a)

    assert ran is True
    assert registry.has(out_key)
    result = registry.get(out_key).collect()
    assert_frame_equal(result, pl.DataFrame({"a": [2, 4, 6]}))
    # The virtual paths must not have been created on disk.
    assert not in_key.parent.exists()
    assert not out_key.parent.exists()


def test_two_stages_chained_in_memory():
    """Stage B reads stage A's output key from the registry — no parquet between them."""
    registry = FrameRegistry()
    in_key = Path("/virtual/in/shard_0.parquet")
    mid_key = Path("/virtual/stage_a/shard_0.parquet")
    out_key = Path("/virtual/stage_b/shard_0.parquet")
    registry.put(in_key, pl.LazyFrame({"a": [-1, 0, 1, 2]}))

    with in_memory_mode(registry):
        rwlock_wrap(in_key, mid_key, read_df, write_df, compute_fn=_double_a)
        rwlock_wrap(mid_key, out_key, read_df, write_df, compute_fn=_filter_positive)

    result = registry.get(out_key).collect()
    # _double_a: [-2, 0, 2, 4] → _filter_positive: [2, 4]
    assert_frame_equal(result, pl.DataFrame({"a": [2, 4]}))


def test_in_memory_matches_on_disk():
    """Same compute sequence gives the same result on disk and in memory."""
    source = pl.DataFrame({"a": [-3, -1, 0, 2, 5]})

    # Disk path: write parquet, run two stages, read back.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fp = tmp / "in.parquet"
        mid_fp = tmp / "mid.parquet"
        out_fp = tmp / "out.parquet"
        source.write_parquet(in_fp)
        rwlock_wrap(in_fp, mid_fp, read_df, write_df, compute_fn=_double_a)
        rwlock_wrap(mid_fp, out_fp, read_df, write_df, compute_fn=_filter_positive)
        disk_result = pl.read_parquet(out_fp)

    # In-memory path: seed a registry with the same source, run the same sequence.
    registry = FrameRegistry()
    in_key = Path("/virtual/in.parquet")
    mid_key = Path("/virtual/mid.parquet")
    out_key = Path("/virtual/out.parquet")
    registry.put(in_key, source.lazy())
    with in_memory_mode(registry):
        rwlock_wrap(in_key, mid_key, read_df, write_df, compute_fn=_double_a)
        rwlock_wrap(mid_key, out_key, read_df, write_df, compute_fn=_filter_positive)
    mem_result = registry.get(out_key).collect()

    assert_frame_equal(disk_result, mem_result)


def test_disk_mode_unaffected_when_no_registry_active():
    """Without in_memory_mode, rwlock_wrap behaves exactly as before (parquet read/write)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fp = tmp / "in.parquet"
        out_fp = tmp / "out.parquet"
        pl.DataFrame({"a": [1, 2]}).write_parquet(in_fp)

        rwlock_wrap(in_fp, out_fp, read_df, write_df, compute_fn=_double_a)

        assert out_fp.is_file()
        assert_frame_equal(pl.read_parquet(out_fp), pl.DataFrame({"a": [2, 4]}))
