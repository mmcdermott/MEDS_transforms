"""End-to-end tests for in-memory execution mode (Phase 0).

These tests verify that:

- A single ``rwlock_wrap`` call under ``in_memory_mode`` reads from and writes to the registry
  without touching the filesystem.
- Two MAP stages can be chained in-memory by setting the second stage's ``input`` key to the
  first stage's ``output`` key.
- The in-memory result is value-equal to the on-disk result computed from the same inputs.
- Disk semantics (``skip-if-exists``, ``do_overwrite``) carry over to the in-memory fast path.
- Stages with custom ``read_fn``/``write_fn`` keep disk-mode locking behavior.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from filelock import FileLock
from polars.testing import assert_frame_equal

from MEDS_transforms.compute_modes import FrameRegistry, in_memory_mode
from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap


def _double_a(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(pl.col("a") * 2)


def _filter_positive(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(pl.col("a") > 0)


def _virtual(tmp_path: Path, *parts: str) -> Path:
    """Return a unique-per-test key that never resolves to a real file."""
    return tmp_path / "virtual" / Path(*parts)


def test_single_stage_in_memory_bypasses_disk(tmp_path: Path):
    """rwlock_wrap under in_memory_mode reads/writes the registry, not the filesystem."""
    registry = FrameRegistry()
    in_key = _virtual(tmp_path, "in", "shard_0.parquet")
    out_key = _virtual(tmp_path, "out", "shard_0.parquet")
    registry.put(in_key, pl.LazyFrame({"a": [1, 2, 3]}))

    with in_memory_mode(registry):
        ran = rwlock_wrap(in_key, out_key, read_df, write_df, compute_fn=_double_a)

    assert ran is True
    assert registry.has(out_key)
    result = registry.get(out_key).collect()
    assert_frame_equal(result, pl.DataFrame({"a": [2, 4, 6]}))
    # The virtual parents must not have been created on disk.
    assert not in_key.parent.exists()
    assert not out_key.parent.exists()


def test_two_stages_chained_in_memory(tmp_path: Path):
    """Stage B reads stage A's output key from the registry — no parquet between them."""
    registry = FrameRegistry()
    in_key = _virtual(tmp_path, "in", "shard_0.parquet")
    mid_key = _virtual(tmp_path, "stage_a", "shard_0.parquet")
    out_key = _virtual(tmp_path, "stage_b", "shard_0.parquet")
    registry.put(in_key, pl.LazyFrame({"a": [-1, 0, 1, 2]}))

    with in_memory_mode(registry):
        rwlock_wrap(in_key, mid_key, read_df, write_df, compute_fn=_double_a)
        rwlock_wrap(mid_key, out_key, read_df, write_df, compute_fn=_filter_positive)

    result = registry.get(out_key).collect()
    # _double_a: [-2, 0, 2, 4] → _filter_positive: [2, 4]
    assert_frame_equal(result, pl.DataFrame({"a": [2, 4]}))


def test_eager_dataframe_chains_like_lazy(tmp_path: Path):
    """A MAP stage that returns an eager DataFrame chains into the next stage cleanly.

    Disk mode returns a ``LazyFrame`` from ``pl.scan_parquet``; the registry must normalize
    eager outputs to lazy so a downstream stage sees the same type regardless of mode.
    """

    def collect_double_a(df: pl.LazyFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("a") * 2).collect()

    registry = FrameRegistry()
    in_key = _virtual(tmp_path, "in.parquet")
    mid_key = _virtual(tmp_path, "mid.parquet")
    out_key = _virtual(tmp_path, "out.parquet")
    registry.put(in_key, pl.LazyFrame({"a": [-1, 0, 1, 2]}))

    with in_memory_mode(registry):
        rwlock_wrap(in_key, mid_key, read_df, write_df, compute_fn=collect_double_a)
        # Second stage calls .filter(), which exists on both DataFrame and LazyFrame but returns
        # different types — the test passes only if the registry normalized mid to LazyFrame.
        mid = registry.get(mid_key)
        assert isinstance(mid, pl.LazyFrame)
        rwlock_wrap(mid_key, out_key, read_df, write_df, compute_fn=_filter_positive)

    result = registry.get(out_key).collect()
    assert_frame_equal(result, pl.DataFrame({"a": [2, 4]}))


def test_in_memory_matches_on_disk(tmp_path: Path):
    """Same compute sequence gives the same result on disk and in memory."""
    source = pl.DataFrame({"a": [-3, -1, 0, 2, 5]})

    in_fp = tmp_path / "in.parquet"
    mid_fp = tmp_path / "mid.parquet"
    out_fp = tmp_path / "out.parquet"
    source.write_parquet(in_fp)
    rwlock_wrap(in_fp, mid_fp, read_df, write_df, compute_fn=_double_a)
    rwlock_wrap(mid_fp, out_fp, read_df, write_df, compute_fn=_filter_positive)
    disk_result = pl.read_parquet(out_fp)

    registry = FrameRegistry()
    in_key = _virtual(tmp_path, "in.parquet")
    mid_key = _virtual(tmp_path, "mid.parquet")
    out_key = _virtual(tmp_path, "out.parquet")
    registry.put(in_key, source.lazy())
    with in_memory_mode(registry):
        rwlock_wrap(in_key, mid_key, read_df, write_df, compute_fn=_double_a)
        rwlock_wrap(mid_key, out_key, read_df, write_df, compute_fn=_filter_positive)
    mem_result = registry.get(out_key).collect()

    assert_frame_equal(disk_result, mem_result)


def test_disk_mode_unaffected_when_no_registry_active(tmp_path: Path):
    """Without in_memory_mode, rwlock_wrap behaves exactly as before (parquet read/write)."""
    in_fp = tmp_path / "in.parquet"
    out_fp = tmp_path / "out.parquet"
    pl.DataFrame({"a": [1, 2]}).write_parquet(in_fp)

    rwlock_wrap(in_fp, out_fp, read_df, write_df, compute_fn=_double_a)

    assert out_fp.is_file()
    assert_frame_equal(pl.read_parquet(out_fp), pl.DataFrame({"a": [2, 4]}))


def test_in_memory_cache_skip_and_overwrite(tmp_path: Path):
    """Fast path honors skip-if-exists and do_overwrite, matching disk-mode semantics."""
    registry = FrameRegistry()
    in_key = _virtual(tmp_path, "in.parquet")
    out_key = _virtual(tmp_path, "out.parquet")
    registry.put(in_key, pl.LazyFrame({"a": [1, 2, 3]}))

    calls = {"n": 0}

    def tracking_double(df: pl.LazyFrame) -> pl.LazyFrame:
        calls["n"] += 1
        return df.with_columns(pl.col("a") * 2)

    with in_memory_mode(registry):
        ran_1 = rwlock_wrap(in_key, out_key, read_df, write_df, compute_fn=tracking_double)
        ran_2 = rwlock_wrap(in_key, out_key, read_df, write_df, compute_fn=tracking_double)
        ran_3 = rwlock_wrap(in_key, out_key, read_df, write_df, compute_fn=tracking_double, do_overwrite=True)

    assert ran_1 is True  # first run: no cached output
    assert ran_2 is False  # second run: cache hit, compute skipped
    assert ran_3 is True  # third run: do_overwrite=True, recomputes
    assert calls["n"] == 2


def test_custom_io_stages_keep_disk_locking(tmp_path: Path):
    """Stages with custom read_fn/write_fn stay on the disk-locking path under in_memory_mode.

    Proves the routing by pre-acquiring the adjacent ``.lock`` file: disk mode observes the
    held lock and returns ``False`` without calling ``csv_write``. If the in-memory fast path
    were (wrongly) taken for custom IO, it would skip the ``FileLock`` and write the CSV —
    which the final assertion catches.
    """
    in_fp = tmp_path / "in.csv"
    out_fp = tmp_path / "out.csv"
    pl.DataFrame({"a": [1, 2]}).write_csv(in_fp)

    write_calls = {"n": 0}

    def csv_read(fp: Path) -> pl.DataFrame:
        return pl.read_csv(fp)

    def csv_write(df: pl.DataFrame, fp: Path) -> None:
        write_calls["n"] += 1
        df.write_csv(fp)

    lock_fp = out_fp.with_suffix(out_fp.suffix + ".lock")
    with FileLock(str(lock_fp)), in_memory_mode():
        ran = rwlock_wrap(in_fp, out_fp, csv_read, csv_write, compute_fn=_double_a)

    assert ran is False, "rwlock_wrap must observe the held lock and return False (disk path)"
    assert write_calls["n"] == 0, "csv_write must not be called when the lock is held"
    assert not out_fp.exists(), "no output should be produced when the disk lock blocks the run"


def test_in_memory_fast_path_serializes_same_out_fp(tmp_path: Path):
    """Concurrent workers targeting the same out_fp serialize like disk mode's FileLock.

    The second caller sees the reservation on ``out_fp`` and returns ``False`` without
    running ``compute_fn`` — mirroring ``FileLock.acquire(timeout=0) → return False`` on disk.
    """
    registry = FrameRegistry()
    in_key = _virtual(tmp_path, "in.parquet")
    out_key = _virtual(tmp_path, "out.parquet")
    registry.put(in_key, pl.LazyFrame({"a": [1, 2]}))
    registry.try_reserve_write(out_key)  # simulate another worker already holding it

    compute_calls = {"n": 0}

    def counting_double(df: pl.LazyFrame) -> pl.LazyFrame:
        compute_calls["n"] += 1
        return df.with_columns(pl.col("a") * 2)

    with in_memory_mode(registry):
        ran = rwlock_wrap(in_key, out_key, read_df, write_df, compute_fn=counting_double)

    assert ran is False
    assert compute_calls["n"] == 0
    assert not registry.has(out_key)

    registry.release_write(out_key)

    with in_memory_mode(registry):
        ran = rwlock_wrap(in_key, out_key, read_df, write_df, compute_fn=counting_double)

    assert ran is True
    assert compute_calls["n"] == 1
    assert_frame_equal(registry.get(out_key).collect(), pl.DataFrame({"a": [2, 4]}))


def test_registry_get_missing_key_has_informative_error(tmp_path: Path):
    """A missing-key read surfaces a registry-scoped KeyError, not a bare dict one."""
    registry = FrameRegistry()
    registry.put(_virtual(tmp_path, "seeded.parquet"), pl.LazyFrame({"a": [1]}))
    missing = _virtual(tmp_path, "absent.parquet")

    with pytest.raises(KeyError, match="In-memory FrameRegistry has no frame registered"):
        registry.get(missing)
